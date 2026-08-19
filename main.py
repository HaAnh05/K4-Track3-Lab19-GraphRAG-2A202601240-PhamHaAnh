import sys
import io

# Đảm bảo Windows console in UTF-8 không bị lỗi charmap
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
from pathlib import Path
import pandas as pd

from src.config import (
    DATA_PATH,
    GOLDEN_PATH,
    OUTPUT_DIR,
    EXTRACTION_MAX_CHUNKS,
)
from src.neo4j_client import connect_neo4j, setup_graph_schema
from src.preprocessing import (
    load_news,
    standardize_news,
    build_chunks,
    run_coref,
)
from src.extraction import (
    run_extraction,
    build_resolution_map,
    canonicalize_triples,
    build_nodes,
    bulk_insert_nodes,
    bulk_insert_edges,
    graph_checks,
)
from src.retrieval import (
    build_flat_index,
    build_entity_matcher,
    answer_flat_rag,
    answer_graph_rag,
)
from src.evaluation import (
    validate_golden,
    run_evaluation,
    comparison_table,
)
from src.bonus import (
    test_supernode_policy,
    show_resolution_audit,
    build_communities,
)

def step_preprocess():
    print("\n" + "="*50)
    print("🚀 BƯỚC 1: TIỀN XỬ LÝ & CHUNKING & COREFERENCE RESOLUTION")
    print("="*50)
    
    print(f"Loading data from: {DATA_PATH}")
    raw_df = load_news(DATA_PATH)
    news_df = standardize_news(raw_df)
    chunks_df = build_chunks(news_df)
    print(f"✅ Đã tạo {len(chunks_df):,} chunks.")
    
    chunks_file = OUTPUT_DIR / "chunks.csv"
    chunks_df.to_csv(chunks_file, index=False)
    
    # Chạy Coreference Resolution trên subset
    extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS).copy()
    coref_file = OUTPUT_DIR / "coref_subset.csv"
    if coref_file.exists():
        print(f"Found cached coref subset at {coref_file}")
        coref_df = pd.read_csv(coref_file)
    else:
        print(f"Running Coreference Resolution on {len(extraction_source)} chunks...")
        coref_df = run_coref(extraction_source)
        coref_df.to_csv(coref_file, index=False)
        print(f"✅ Đã lưu {len(coref_df)} coref chunks vào {coref_file}.")
        
    return chunks_df, coref_df

def step_extract(coref_df=None):
    print("\n" + "="*50)
    print("🚀 BƯỚC 2: TRÍCH XUẤT QUAN HỆ & ENTITY RESOLUTION")
    print("="*50)
    
    triples_file = OUTPUT_DIR / "triples_raw.csv"
    if triples_file.exists():
        print(f"Found cached triples at {triples_file}")
        raw_triples_df = pd.read_csv(triples_file)
    else:
        if coref_df is None:
            coref_file = OUTPUT_DIR / "coref_subset.csv"
            if coref_file.exists():
                coref_df = pd.read_csv(coref_file)
            else:
                _, coref_df = step_preprocess()
        print(f"Running NER + RE Extraction on {len(coref_df)} chunks...")
        raw_triples_df, _ = run_extraction(coref_df)
        raw_triples_df.to_csv(triples_file, index=False)
        print(f"✅ Đã trích xuất {len(raw_triples_df)} raw triples.")

    print("Running Entity Resolution (Vector ANN + Lexical Guard + Union-Find)...")
    entity_map, audit_df = build_resolution_map(raw_triples_df)
    triples_df = canonicalize_triples(raw_triples_df, entity_map)
    nodes_df = build_nodes(triples_df)
    
    show_resolution_audit(audit_df)
    print(f"✅ Chuẩn hóa thành công: {len(nodes_df)} nodes và {len(triples_df)} edges.")
    return nodes_df, triples_df

def step_ingest(nodes_df=None, triples_df=None):
    print("\n" + "="*50)
    print("🚀 BƯỚC 3: BULK INGESTION VÀO NEO4J & INTEGRITY CHECK")
    print("="*50)
    
    if nodes_df is None or triples_df is None:
        nodes_df, triples_df = step_extract()
        
    connect_neo4j()
    setup_graph_schema()
    
    print("Bulk inserting nodes...")
    bulk_insert_nodes(nodes_df)
    print("Bulk inserting edges with full provenance...")
    bulk_insert_edges(triples_df)
    
    counts, top_degree_df = graph_checks()
    print("✅ Ingestion hoàn tất. Top Entities theo Degree:")
    print(top_degree_df)
    return counts

def step_eval():
    print("\n" + "="*50)
    print("🚀 BƯỚC 4: BENCHMARK ĐÁNH GIÁ LLM-AS-A-JUDGE")
    print("="*50)
    
    # Đảm bảo Flat index và Entity Matcher được nạp
    chunks_file = OUTPUT_DIR / "chunks.csv"
    if chunks_file.exists():
        chunks_df = pd.read_csv(chunks_file)
    else:
        chunks_df, _ = step_preprocess()
    build_flat_index(chunks_df)
    
    triples_file = OUTPUT_DIR / "triples_raw.csv"
    if triples_file.exists():
        raw_triples = pd.read_csv(triples_file)
        entity_map, _ = build_resolution_map(raw_triples)
        triples_df = canonicalize_triples(raw_triples, entity_map)
        nodes_df = build_nodes(triples_df)
        build_entity_matcher(nodes_df)
        
    print(f"Loading Golden Dataset from: {GOLDEN_PATH}")
    golden_df = pd.read_csv(GOLDEN_PATH)
    validate_golden(golden_df, require_answers=True)
    
    print(f"Evaluating {len(golden_df)} questions...")
    eval_results_df = run_evaluation(golden_df)
    
    comparison_df = comparison_table(eval_results_df)
    print("\n=== BẢNG SO SÁNH TỔNG HỢP (SUMMARY) ===")
    print(comparison_df)
    
    res_path = OUTPUT_DIR / "graphrag_eval_results.csv"
    sum_path = OUTPUT_DIR / "graphrag_vs_flatrag_summary.csv"
    eval_results_df.to_csv(res_path, index=False)
    comparison_df.to_csv(sum_path, index=False)
    print(f"\n✅ Đã lưu kết quả đánh giá vào {res_path} và {sum_path}!")

def step_bonus():
    print("\n" + "="*50)
    print("🚀 BƯỚC 5: SUPER-NODE CHECK & COMMUNITY DETECTION")
    print("="*50)
    
    test_supernode_policy()
    comm_df = build_communities()
    print(f"✅ Bonus checks hoàn tất. Đã phân cụm {len(comm_df)} nodes.")

def step_query(question: str):
    print("\n" + "="*50)
    print(f"❓ TRUY VẤN THỬ NGHIỆM: {question}")
    print("="*50)
    
    chunks_file = OUTPUT_DIR / "chunks.csv"
    if chunks_file.exists():
        chunks_df = pd.read_csv(chunks_file)
        build_flat_index(chunks_df)
    
    triples_file = OUTPUT_DIR / "triples_raw.csv"
    if triples_file.exists():
        raw_triples = pd.read_csv(triples_file)
        entity_map, _ = build_resolution_map(raw_triples)
        triples_df = canonicalize_triples(raw_triples, entity_map)
        nodes_df = build_nodes(triples_df)
        build_entity_matcher(nodes_df)

    print("\n--- 1. FLAT RAG ANSWER ---")
    flat_out = answer_flat_rag(question)
    print(f"Answer: {flat_out['answer']}")
    print(f"Latency: {flat_out['latency_s']:.2f}s | Tokens: {flat_out['total_tokens']}")
    
    print("\n--- 2. HYBRID GRAPHRAG ANSWER ---")
    graph_out = answer_graph_rag(question)
    print(f"Answer: {graph_out['answer']}")
    print(f"Latency: {graph_out['latency_s']:.2f}s | Tokens: {graph_out['total_tokens']}")

def main():
    parser = argparse.ArgumentParser(description="Production-Grade GraphRAG vs Flat RAG Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "preprocess", "extract", "ingest", "eval", "bonus"],
        default="all",
        help="Chọn bước cần thực thi (mặc định: all)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Chạy thử nghiệm câu hỏi cụ thể so sánh Flat RAG vs GraphRAG",
    )
    args = parser.parse_args()

    if args.query:
        step_query(args.query)
        return

    if args.step == "all":
        chunks_df, coref_df = step_preprocess()
        nodes_df, triples_df = step_extract(coref_df)
        step_ingest(nodes_df, triples_df)
        step_eval()
        step_bonus()
        print("\n🎉 HOÀN THÀNH TOÀN BỘ PIPELINE!")
    elif args.step == "preprocess":
        step_preprocess()
    elif args.step == "extract":
        step_extract()
    elif args.step == "ingest":
        step_ingest()
    elif args.step == "eval":
        step_eval()
    elif args.step == "bonus":
        step_bonus()

if __name__ == "__main__":
    main()
