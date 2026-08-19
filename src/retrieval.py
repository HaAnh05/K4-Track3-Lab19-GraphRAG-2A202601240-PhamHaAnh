import time
from collections import deque
import faiss
import numpy as np
import pandas as pd

from src.config import (
    ALLOWED_NODE_TYPES,
    GROQ_MODEL,
    SUPER_NODE_DEGREE,
    SUPER_NODE_EDGE_CAP,
    GLOBAL_EDGE_CAP,
    MAX_GRAPH_CONTEXT_CHARS,
    get_embedder,
)
from src.llm_client import groq_chat, groq_json
from src.neo4j_client import run_cypher
from src.preprocessing import norm_space
from src.extraction import norm_entity

# --- FAISS Flat Vector Index Store ---
flat_index = None
flat_store = None

def build_flat_index(chunks_df: pd.DataFrame):
    """
    Xây dựng FAISS Vector Index cho Flat RAG.
    """
    global flat_index, flat_store
    vecs = get_embedder().encode(
        chunks_df.text.fillna("").tolist(),
        batch_size=128, show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    flat_index = faiss.IndexFlatIP(vecs.shape[1])
    flat_index.add(vecs)
    flat_store = chunks_df.reset_index(drop=True).copy()
    print(f"✅ Flat Index ready with {flat_index.ntotal:,} vectors.")

def retrieve_flat_context(query: str, k: int = 6) -> tuple[str, pd.DataFrame]:
    """
    Truy vấn top-k chunk gần nhất theo cosine similarity.
    """
    global flat_index, flat_store
    if flat_index is None:
        raise RuntimeError("Flat Index chưa được xây dựng. Vui lòng gọi build_flat_index(chunks_df) trước.")

    qv = get_embedder().encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")
    scores, ids = flat_index.search(qv, min(k, flat_index.ntotal))

    rows = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        r = flat_store.iloc[int(idx)]
        rows.append({
            "score": float(score), "chunk_id": r.chunk_id,
            "published_date": r.published_date, "text": r.text
        })

    df = pd.DataFrame(rows)
    context = "\n\n".join(
        f"[chunk_id={r.chunk_id} | date={r.published_date} | score={r.score:.3f}]\n{r.text}"
        for r in df.itertuples(index=False)
    )
    return context, df

# --- Seed Entity Extraction & Matching ---
SEED_SYSTEM = """
Extract useful seed entities for graph retrieval.
Allowed types: Company, Person, Technology.
Do not answer the question. Return strict JSON only.
""".strip()

def extract_seeds(query: str) -> list:
    """Trích xuất danh sách thực thể hạt nhân từ câu hỏi qua LLM."""
    obj, _ = groq_json(SEED_SYSTEM, f"""
Question: {query}
Return {{"seeds":[{{"name":"...","type":"Company|Person|Technology|null"}}]}}
""")
    return [
        {"name": norm_space(x.get("name")),
         "type": x.get("type") if x.get("type") in ALLOWED_NODE_TYPES else None}
        for x in obj.get("seeds", [])
        if norm_space(x.get("name"))
    ]

entity_match_vectors = None
entity_match_store = None

def build_entity_matcher(nodes_df: pd.DataFrame):
    """
    Xây dựng vector index hỗ trợ Fuzzy Match cho Seed Entities.
    """
    global entity_match_vectors, entity_match_store
    entity_match_store = nodes_df.reset_index(drop=True).copy()
    entity_match_vectors = get_embedder().encode(
        entity_match_store.name.tolist(),
        batch_size=128, show_progress_bar=False,
        normalize_embeddings=True
    ).astype("float32")
    print(f"✅ Entity Matcher ready with {len(entity_match_store):,} entities.")

def match_seeds(query: str, fuzzy_threshold: float = 0.66) -> list:
    """
    Khớp seed entity với node trong Neo4j (Exact match + Fallback vector).
    """
    global entity_match_vectors, entity_match_store
    matched = []
    for seed in extract_seeds(query):
        exact = run_cypher("""
        MATCH (n:Entity)
        WHERE (n.name_norm=$name OR $name IN coalesce(n.aliases_norm,[]))
          AND ($typ IS NULL OR n.entity_type=$typ)
        RETURN n.id AS id, n.name AS name, n.entity_type AS type
        LIMIT 5
        """, name=norm_entity(seed["name"]), typ=seed["type"])

        if exact:
            matched += exact
            continue

        if entity_match_vectors is None:
            continue

        mask = np.ones(len(entity_match_store), dtype=bool)
        if seed["type"]:
            mask = entity_match_store.type.eq(seed["type"]).to_numpy()
        idxs = np.flatnonzero(mask)
        if not len(idxs):
            continue

        qv = get_embedder().encode(
            [seed["name"]], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")[0]
        sims = entity_match_vectors[idxs] @ qv
        j = int(np.argmax(sims))
        if float(sims[j]) >= fuzzy_threshold:
            r = entity_match_store.iloc[int(idxs[j])]
            matched.append({"id": r.id, "name": r.name, "type": r.type})

    return list({x["id"]: x for x in matched}.values())

# --- Graph Traversal & Super-node Mitigation ---
def node_degree(node_id: str) -> int:
    """Đếm bậc của một node trong Neo4j."""
    return int(run_cypher("""
    MATCH (n:Entity {id:$id})
    OPTIONAL MATCH (n)-[r]-()
    RETURN count(r) AS degree
    """, id=node_id)[0]["degree"])

def recent_edges(node_id: str, limit: int) -> list:
    """Lấy các cạnh nối với node_id ưu tiên ngày mới nhất."""
    return run_cypher("""
    MATCH (n:Entity {id:$id})
    MATCH (n)-[r]-(m:Entity)
    RETURN
      startNode(r).id AS source_id,
      startNode(r).name AS source_name,
      startNode(r).entity_type AS source_type,
      type(r) AS relation,
      endNode(r).id AS target_id,
      endNode(r).name AS target_name,
      endNode(r).entity_type AS target_type,
      r.source_chunk_id AS source_chunk_id,
      r.published_date AS published_date,
      r.evidence AS evidence,
      m.id AS neighbor_id
    ORDER BY coalesce(r.published_date,'') DESC
    LIMIT $limit
    """, id=node_id, limit=int(limit))

def textualize(edges: list) -> str:
    """Linearize danh sách cạnh đồ thị thành dạng text có cấu trúc."""
    edges = sorted(edges, key=lambda e: e.get("published_date") or "", reverse=True)
    lines, used = [], 0
    for e in edges:
        line = (
            f"{e['source_name']} [{e['source_type']}] -{e['relation']}-> "
            f"{e['target_name']} [{e['target_type']}] "
            f"| date={e.get('published_date') or 'unknown'} "
            f"| chunk={e.get('source_chunk_id') or 'unknown'}"
        )
        if e.get("evidence"):
            line += f" | evidence={norm_space(e['evidence'])}"
        if used + len(line) + 1 > MAX_GRAPH_CONTEXT_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)

def retrieve_graph_context(query: str, max_hops: int = 2, edge_limit: int = 50, return_debug: bool = False):
    """
    Truy vấn đồ thị theo BFS có kiểm soát Super-node degree cap.
    """
    seeds = match_seeds(query)
    if not seeds:
        out = {
            "context": "", "edges": pd.DataFrame(),
            "diagnostics": {"reason": "NO_SEED", "supernode_events": []}
        }
        return out if return_debug else ""

    frontier = deque((x["id"], 0) for x in seeds)
    expanded, seen_edges, collected = set(), set(), []
    supernode_events = []

    while frontier and len(collected) < GLOBAL_EDGE_CAP:
        node_id, hop = frontier.popleft()
        if node_id in expanded or hop >= max_hops:
            continue
        expanded.add(node_id)

        degree = node_degree(node_id)
        limit = int(edge_limit)
        if degree > SUPER_NODE_DEGREE:
            limit = min(limit, SUPER_NODE_EDGE_CAP)
            supernode_events.append({"node_id": node_id, "degree": degree, "limit": limit})

        for e in recent_edges(node_id, limit):
            key = (e["source_id"], e["relation"], e["target_id"], e["source_chunk_id"])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            collected.append(e)
            if len(collected) >= GLOBAL_EDGE_CAP:
                break

            nb = e.get("neighbor_id")
            if nb and nb not in expanded and hop + 1 < max_hops:
                frontier.append((nb, hop + 1))

    out = {
        "context": textualize(collected),
        "edges": pd.DataFrame(collected),
        "diagnostics": {
            "matched_seeds": seeds,
            "expanded_nodes": len(expanded),
            "collected_edges": len(collected),
            "supernode_events": supernode_events,
        }
    }
    return out if return_debug else out["context"]

# --- Answer Generation ---
ANSWER_SYSTEM = """
Answer only from supplied context.
Be concise but complete. Do not invent facts.
Cite provenance inline as [chunk_id=...] whenever possible.
If evidence is insufficient or conflicting, say so.
""".strip()

def generate_answer(question: str, context: str) -> dict:
    """Sinh câu trả lời dựa trên context được cung cấp."""
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:"
    t0 = time.perf_counter()
    text, usage = groq_chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        model=GROQ_MODEL
    )
    return {
        "answer": text.strip(),
        "latency_s": time.perf_counter() - t0,
        "total_tokens": usage.get("total_tokens", 0),
    }

def answer_flat_rag(question: str) -> dict:
    """Pipeline trả lời của Flat RAG thuần túy."""
    context, retrieved = retrieve_flat_context(question, k=6)
    out = generate_answer(question, context)
    out.update({"context": context, "retrieved": retrieved})
    return out

def answer_graph_rag(question: str) -> dict:
    """Pipeline trả lời của Hybrid GraphRAG (Subgraph + Vector chunks)."""
    g = retrieve_graph_context(question, max_hops=2, edge_limit=50, return_debug=True)
    vctx, vdocs = retrieve_flat_context(question, k=4)
    context = f"=== GRAPH ===\n{g['context']}\n\n=== VECTOR ===\n{vctx}"
    out = generate_answer(question, context)
    out.update({"context": context, "graph_debug": g, "vector_docs": vdocs})
    return out
