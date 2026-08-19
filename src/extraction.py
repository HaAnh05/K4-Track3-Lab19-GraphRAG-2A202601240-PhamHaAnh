import re
import json
import unicodedata
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import (
    ALLOWED_NODE_TYPES,
    ALLOWED_RELATIONS,
    get_embedder,
    OUTPUT_DIR,
)
from src.llm_client import groq_json
from src.neo4j_client import run_cypher
from src.preprocessing import norm_space, sha1

EXTRACT_SYSTEM = f"""
Extract a high-precision knowledge graph from tech-news text.
Allowed node types: {sorted(ALLOWED_NODE_TYPES)}
Allowed relations: {sorted(ALLOWED_RELATIONS)}
Use only explicitly supported facts. Prefer precision over recall.
Every relation needs short evidence. Return strict JSON only.
""".strip()

def extract_batch(batch_df: pd.DataFrame):
    """
    Trích xuất quan hệ triples (Source, Relation, Target) từ batch chunks qua LLM.
    """
    payload = [{
        "chunk_id": r.chunk_id,
        "published_date": r.published_date,
        "text": getattr(r, "resolved_text", None) or r.text,
    } for r in batch_df.itertuples(index=False)]

    prompt = f"""
Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "relations": [
        {{
          "source": "...",
          "source_type": "Company|Person|Technology",
          "relation": "ALLOWED_RELATION",
          "target": "...",
          "target_type": "Company|Person|Technology",
          "evidence": "...",
          "confidence": 0.0
        }}
      ]
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
    return groq_json(EXTRACT_SYSTEM, prompt)

def run_extraction(source_df: pd.DataFrame, batch_size: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chạy trích xuất toàn bộ quan hệ từ source DataFrame.
    """
    meta = source_df.set_index("chunk_id")["published_date"].to_dict()
    triples, errors = [], []

    for start in tqdm(range(0, len(source_df), batch_size), desc="NER+RE"):
        batch = source_df.iloc[start:start+batch_size]
        try:
            obj, _ = extract_batch(batch)
        except Exception as e:
            errors.append({"start": start, "error": str(e)})
            continue

        for item in obj.get("items", []):
            cid = item.get("chunk_id")
            if cid not in meta:
                continue
            for x in item.get("relations", []):
                s = norm_space(x.get("source"))
                t = norm_space(x.get("target"))
                st = x.get("source_type")
                tt = x.get("target_type")
                rel = x.get("relation")
                
                if not s or not t:
                    continue
                if st not in ALLOWED_NODE_TYPES or tt not in ALLOWED_NODE_TYPES:
                    continue
                if rel not in ALLOWED_RELATIONS:
                    continue
                
                triples.append({
                    "source_raw": s,
                    "source_type": st,
                    "relation": rel,
                    "target_raw": t,
                    "target_type": tt,
                    "source_chunk_id": cid,
                    "published_date": meta[cid] or "",
                    "evidence": norm_space(x.get("evidence")),
                    "confidence": float(x.get("confidence") or 0.0),
                })

    return pd.DataFrame(triples), pd.DataFrame(errors)

# --- Entity Resolution ---
CORP_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc", "plc", "co", "company"}
MANUAL_ALIASES = {
    "msft": "Microsoft",
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "goog": "Google",
    "googl": "Google",
    "google llc": "Google",
    "meta platforms": "Meta",
    "meta platforms inc": "Meta",
    "aapl": "Apple",
    "apple inc": "Apple",
}

def norm_entity(name: str) -> str:
    """Chuẩn hóa tên thực thể về dạng chữ thường không dấu đặc biệt."""
    s = unicodedata.normalize("NFKC", norm_space(name)).lower()
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def strip_suffix(name: str) -> str:
    """Loại bỏ các hậu tố công ty như Inc, Corp, LLC."""
    toks = norm_entity(name).replace(".", "").split()
    while toks and toks[-1] in CORP_SUFFIXES:
        toks.pop()
    return " ".join(toks)

def merge_guard(a: str, b: str) -> bool:
    """Lexical Guard ngăn chặn False Merge nguy hiểm."""
    na, nb = strip_suffix(a), strip_suffix(b)
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.72

class UF:
    """Disjoint-Set Union (Union-Find) cho phân cụm thực thể."""
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a: int, b: int):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a

def build_resolution_map(raw_triples_df: pd.DataFrame, threshold: float = 0.90, top_k: int = 5) -> tuple[dict, pd.DataFrame]:
    """
    Xây dựng từ điển ánh xạ thực thể đồng nghĩa kết hợp Vector ANN + Lexical Guard + Union-Find.
    """
    mentions = []
    for r in raw_triples_df.itertuples(index=False):
        mentions += [(r.source_type, r.source_raw), (r.target_type, r.target_raw)]

    counts = Counter((t, norm_entity(n)) for t, n in mentions)
    display_name = {}
    for t, n in mentions:
        display_name.setdefault((t, norm_entity(n)), n)

    mapping, audit = {}, []

    for key in counts:
        t, norm = key
        if norm in MANUAL_ALIASES:
            mapping[key] = MANUAL_ALIASES[norm]
            audit.append({
                "type": t, "left": display_name[key],
                "right": MANUAL_ALIASES[norm],
                "similarity": 1.0, "decision": "MERGE_MANUAL"
            })

    for typ in sorted(ALLOWED_NODE_TYPES):
        keys = [k for k in counts if k[0] == typ and k not in mapping]
        if not keys:
            continue
        names = [display_name[k] for k in keys]
        vecs = get_embedder().encode(
            names, batch_size=128, show_progress_bar=False,
            normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        sims, nbrs = index.search(vecs, min(top_k, len(names)))
        uf = UF(len(names))

        for i in range(len(names)):
            for score, j in zip(sims[i], nbrs[i]):
                if j < 0 or i >= j or float(score) < threshold:
                    continue
                ok = merge_guard(names[i], names[j])
                audit.append({
                    "type": typ, "left": names[i], "right": names[j],
                    "similarity": float(score),
                    "decision": "MERGE_VECTOR" if ok else "REJECT_GUARD"
                })
                if ok:
                    uf.union(i, j)

        groups = defaultdict(list)
        for i in range(len(names)):
            groups[uf.find(i)].append(i)

        for idxs in groups.values():
            best = sorted(
                idxs,
                key=lambda i: (-counts[keys[i]], len(names[i]), names[i].lower())
            )[0]
            canonical = names[best]
            for i in idxs:
                mapping[keys[i]] = canonical

    for key in counts:
        mapping.setdefault(key, display_name[key])

    return mapping, pd.DataFrame(audit)

def canonicalize_triples(raw_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Chuẩn hóa quan hệ triples sang ID và tên thực thể chuẩn hóa.
    """
    df = raw_df.copy()
    def canon(name, typ):
        n = norm_entity(name)
        return mapping.get((typ, n), MANUAL_ALIASES.get(n, name))

    df["source_name"] = [canon(n, t) for n, t in zip(df.source_raw, df.source_type)]
    df["target_name"] = [canon(n, t) for n, t in zip(df.target_raw, df.target_type)]
    df["source_name_norm"] = df.source_name.map(norm_entity)
    df["target_name_norm"] = df.target_name.map(norm_entity)
    df["source_id"] = [sha1(f"{t}:{n}")[:24] for t, n in zip(df.source_type, df.source_name_norm)]
    df["target_id"] = [sha1(f"{t}:{n}")[:24] for t, n in zip(df.target_type, df.target_name_norm)]
    return df[df.source_id != df.target_id].reset_index(drop=True)

# --- Bulk Ingestion into Neo4j ---
def build_nodes(triples_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng danh sách node độc nhất từ quan hệ triples."""
    rows = []
    for r in triples_df.itertuples(index=False):
        rows += [
            {"id": r.source_id, "name": r.source_name, "name_norm": r.source_name_norm, "type": r.source_type, "alias": r.source_raw},
            {"id": r.target_id, "name": r.target_name, "name_norm": r.target_name_norm, "type": r.target_type, "alias": r.target_raw},
        ]
    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return tmp

    out = []
    for (node_id, name, name_norm, typ), g in tmp.groupby(["id", "name", "name_norm", "type"]):
        aliases = sorted(set(g["alias"].map(norm_space)))
        out.append({
            "id": node_id, "name": name, "name_norm": name_norm, "type": typ,
            "aliases": aliases,
            "aliases_norm": sorted(set(norm_entity(x) for x in aliases))
        })
    return pd.DataFrame(out)

def batches(records: list, size: int = 1000):
    for i in range(0, len(records), size):
        yield records[i:i+size]

def bulk_insert_nodes(nodes_df: pd.DataFrame, batch_size: int = 1000):
    """Nạp nodes vào Neo4j sử dụng UNWIND theo batch."""
    for typ in sorted(ALLOWED_NODE_TYPES):
        part = nodes_df[nodes_df.type == typ]
        if part.empty:
            continue
        query = f"""
        UNWIND $rows AS row
        MERGE (n:Entity {{id: row.id}})
        SET n:{typ},
            n.name=row.name,
            n.name_norm=row.name_norm,
            n.entity_type=row.type,
            n.aliases=row.aliases,
            n.aliases_norm=row.aliases_norm
        """
        for b in batches(part.to_dict("records"), batch_size):
            run_cypher(query, rows=b)

def bulk_insert_edges(triples_df: pd.DataFrame, batch_size: int = 1000):
    """Nạp edges vào Neo4j sử dụng UNWIND theo batch kèm provenance đầy đủ."""
    required = {"source_chunk_id", "published_date"}
    if not required.issubset(triples_df.columns):
        raise ValueError("Missing edge provenance.")

    for rel in sorted(ALLOWED_RELATIONS):
        part = triples_df[triples_df.relation == rel]
        if part.empty:
            continue

        query = f"""
        UNWIND $rows AS row
        MATCH (s:Entity {{id: row.source_id}})
        MATCH (t:Entity {{id: row.target_id}})
        MERGE (s)-[r:{rel} {{source_chunk_id: row.source_chunk_id}}]->(t)
        SET r.published_date=row.published_date,
            r.evidence=row.evidence,
            r.confidence=row.confidence
        """

        cols = ["source_id", "target_id", "source_chunk_id", "published_date", "evidence", "confidence"]
        for b in batches(part[cols].to_dict("records"), batch_size):
            run_cypher(query, rows=b)

def graph_checks() -> tuple[dict, pd.DataFrame]:
    """
    Kiểm tra tính toàn vẹn của đồ thị: 0 cạnh bị thiếu provenance.
    """
    invalid = run_cypher("""
    MATCH ()-[r]->()
    WHERE r.source_chunk_id IS NULL OR r.published_date IS NULL
    RETURN count(r) AS n
    """)[0]["n"]

    counts = {
        "nodes": run_cypher("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"],
        "edges": run_cypher("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"],
        "invalid_provenance_edges": invalid,
    }
    print(f"Graph Integrity Check: {counts}")
    assert invalid == 0, f"Lỗi: Có {invalid} cạnh bị thiếu provenance!"

    top = pd.DataFrame(run_cypher("""
    MATCH (n:Entity)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) AS degree
    RETURN n.id AS id, n.name AS name, n.entity_type AS type, degree
    ORDER BY degree DESC LIMIT 15
    """))
    return counts, top
