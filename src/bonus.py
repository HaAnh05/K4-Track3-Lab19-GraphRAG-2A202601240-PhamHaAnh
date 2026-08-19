from pathlib import Path
import networkx as nx
import pandas as pd

from src.config import SUPER_NODE_DEGREE, OUTPUT_DIR
from src.extraction import batches
from src.llm_client import groq_json
from src.neo4j_client import run_cypher
from src.preprocessing import norm_space
from src.retrieval import recent_edges, retrieve_graph_context, retrieve_flat_context

def test_supernode_policy():
    """
    Kiểm tra chính sách Super-node mitigation: Node bậc > 100 bị cắt tỉa <= 50 cạnh mới nhất.
    """
    rows = run_cypher("""
    MATCH (n:Entity)-[r]-()
    WITH n, count(r) AS degree
    ORDER BY degree DESC LIMIT 1
    RETURN n.id AS id, n.name AS name, degree
    """)
    if not rows:
        print("Graph is empty.")
        return

    n = rows[0]
    limit = 50 if n["degree"] > SUPER_NODE_DEGREE else 1000
    edges = recent_edges(n["id"], limit)
    print(f"Top Super-node: {n} -> Fetched {len(edges)} edges.")
    if n["degree"] > SUPER_NODE_DEGREE:
        assert len(edges) <= 50, f"Lỗi: Số cạnh lấy được ({len(edges)}) vượt quá giới hạn 50!"
        print("✅ Super-node cap OK: Degree > 100 cắt tỉa chính xác về <= 50 edges.")

def show_resolution_audit(audit_df: pd.DataFrame):
    """
    Hiển thị và xuất báo cáo kiểm toán phân giải thực thể.
    """
    if audit_df is None or audit_df.empty:
        print("No audit rows.")
        return
    print("=== Top Entity Resolution Pairs ===")
    print(audit_df.sort_values("similarity", ascending=False).head(15))
    
    out_file = OUTPUT_DIR / "entity_resolution_audit.csv"
    audit_df.to_csv(out_file, index=False)
    print(f"✅ Đã lưu {len(audit_df)} dòng audit vào {out_file}!")

# --- Bonus 1: NetworkX Community Detection Fallback ---
def build_communities(limit_edges: int = 20000) -> pd.DataFrame:
    """
    Phân cụm cộng đồng bằng thuật toán Greedy Modularity của NetworkX và gán community_id vào Neo4j.
    """
    edge_df = pd.DataFrame(run_cypher("""
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.id AS source, b.id AS target
    LIMIT $limit
    """, limit=int(limit_edges)))

    if edge_df.empty:
        print("Không có cạnh nào để phân cụm cộng đồng.")
        return pd.DataFrame()

    G = nx.Graph()
    G.add_edges_from(edge_df[["source", "target"]].itertuples(index=False, name=None))
    communities = nx.algorithms.community.greedy_modularity_communities(G)

    rows = []
    for cid, members in enumerate(communities):
        rows += [{"id": node_id, "community_id": int(cid)} for node_id in members]

    for b in batches(rows, 1000):
        run_cypher("""
        UNWIND $rows AS row
        MATCH (n:Entity {id: row.id})
        SET n.community_id = row.community_id
        """, rows=b)

    comm_df = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / "community_reports.csv"
    comm_df.to_csv(out_file, index=False)
    print(f"✅ Đã phát hiện {len(communities)} cộng đồng và lưu vào {out_file}!")
    return comm_df

# --- Bonus 2: Self-Correction Scaffold ---
SUFFICIENCY_SYSTEM = """
Decide whether the supplied retrieval context is sufficient to answer the question faithfully.
Do not answer the question. Return strict JSON only.
""".strip()

def context_sufficient(question: str, context: str) -> tuple[bool, str]:
    """
    Đánh giá xem ngữ cảnh truy xuất có đủ thông tin để trả lời câu hỏi hay không.
    """
    obj, _ = groq_json(
        SUFFICIENCY_SYSTEM,
        f"""QUESTION: {question}
CONTEXT:
{context[:16000]}
Return {{"sufficient":true,"missing":"..."}}"""
    )
    return bool(obj.get("sufficient")), norm_space(obj.get("missing"))

def self_correcting_context(question: str) -> dict:
    """
    Cơ chế mở rộng đồ thị thích ứng (Adaptive Hop expansion) hoặc fallback sang Hybrid vector.
    """
    g2 = retrieve_graph_context(question, max_hops=2, edge_limit=50, return_debug=True)
    ok, missing = context_sufficient(question, g2["context"])
    if ok:
        return {"route": "hop2", "context": g2["context"], "missing": ""}

    g3 = retrieve_graph_context(question, max_hops=3, edge_limit=50, return_debug=True)
    ok, missing2 = context_sufficient(question, g3["context"])
    if ok:
        return {"route": "hop3", "context": g3["context"], "missing": missing}

    flat, _ = retrieve_flat_context(question, k=8)
    return {
        "route": "hop3+vector",
        "context": f"=== GRAPH ===\n{g3['context']}\n\n=== VECTOR ===\n{flat}",
        "missing": missing2
    }
