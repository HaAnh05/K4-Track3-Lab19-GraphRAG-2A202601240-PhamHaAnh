from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import (
    JUDGE_PROVIDER,
    JUDGE_MODEL,
    OPENAI_API_KEY,
    OUTPUT_DIR,
)
from src.llm_client import groq_json, parse_json_object
from src.preprocessing import norm_space
from src.retrieval import answer_flat_rag, answer_graph_rag

JUDGE_SYSTEM = """
You are a strict evaluator of RAG answers.
Score 1-5:
- comprehensiveness
- faithfulness to supplied candidate context
- multi_hop_reasoning accuracy
Use the reference answer as correctness anchor.
Return strict JSON only.
""".strip()

def validate_golden(df: pd.DataFrame, require_answers: bool = True):
    """
    Kiểm tra cấu trúc và tính hợp lệ của Golden Dataset.
    """
    required = {"id", "group", "question", "reference_answer"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in golden dataset: {required - set(df.columns)}")
    if require_answers and df.reference_answer.fillna("").str.strip().eq("").any():
        empty_rows = df[df.reference_answer.fillna("").str.strip().eq("")][["id", "question"]]
        raise ValueError(f"Có {len(empty_rows)} câu hỏi chưa có reference_answer: \n{empty_rows}")
    print(f"✅ Golden Dataset hợp lệ với {len(df)} câu hỏi.")

def judge_json(system: str, user: str) -> dict:
    """
    Thực hiện gọi LLM Judge qua Groq hoặc OpenAI theo cấu hình.
    """
    if not JUDGE_MODEL:
        raise RuntimeError("Thiếu JUDGE_MODEL.")

    if JUDGE_PROVIDER == "groq":
        return groq_json(system, user, model=JUDGE_MODEL)[0]

    if JUDGE_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("Thiếu OPENAI_API_KEY.")
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return parse_json_object(resp.choices[0].message.content)

    raise ValueError("JUDGE_PROVIDER must be 'openai' or 'groq'.")

def judge_answer(question: str, reference: str, answer: str, context: str) -> dict:
    """
    Chấm điểm câu trả lời của RAG trên 3 tiêu chí: Comprehensiveness, Faithfulness, Multi-hop reasoning.
    """
    prompt = f"""
QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{answer}

CANDIDATE CONTEXT:
{context[:18000]}

Return:
{{
 "comprehensiveness":1,
 "faithfulness":1,
 "multi_hop_reasoning":1,
 "rationale":"2-5 sentences"
}}
"""
    obj = judge_json(JUDGE_SYSTEM, prompt)
    out = {}
    for k in ["comprehensiveness", "faithfulness", "multi_hop_reasoning"]:
        out[k] = max(1, min(5, int(obj.get(k, 1))))
    out["rationale"] = norm_space(obj.get("rationale"))
    return out

def run_evaluation(golden_df: pd.DataFrame, checkpoint_path: str = None) -> pd.DataFrame:
    """
    Chạy đánh giá song song Flat RAG vs GraphRAG trên Golden Dataset qua LLM Judge.
    """
    checkpoint_file = Path(checkpoint_path or (OUTPUT_DIR / "graphrag_eval_checkpoint.csv"))
    rows = []

    for q in tqdm(golden_df.itertuples(index=False), total=len(golden_df), desc="Evaluation"):
        flat = answer_flat_rag(q.question)
        graph = answer_graph_rag(q.question)

        jf = judge_answer(q.question, q.reference_answer, flat["answer"], flat["context"])
        jg = judge_answer(q.question, q.reference_answer, graph["answer"], graph["context"])

        rows.append({
            "id": q.id, "group": q.group, "question": q.question,
            "reference_answer": q.reference_answer,
            "flat_answer": flat["answer"], "graph_answer": graph["answer"],
            "flat_comprehensiveness": jf["comprehensiveness"],
            "graph_comprehensiveness": jg["comprehensiveness"],
            "flat_faithfulness": jf["faithfulness"],
            "graph_faithfulness": jg["faithfulness"],
            "flat_multi_hop_reasoning": jf["multi_hop_reasoning"],
            "graph_multi_hop_reasoning": jg["multi_hop_reasoning"],
            "flat_latency_s": flat["latency_s"],
            "graph_latency_s": graph["latency_s"],
            "flat_total_tokens": flat.get("total_tokens", 0),
            "graph_total_tokens": graph.get("total_tokens", 0),
            "flat_judge_rationale": jf["rationale"],
            "graph_judge_rationale": jg["rationale"],
            "graph_supernode_events": len(
                graph["graph_debug"]["diagnostics"].get("supernode_events", [])
            )
        })
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(checkpoint_file, index=False)

    return pd.DataFrame(rows)

def comparison_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng so sánh tổng hợp chi tiết theo từng nhóm câu hỏi (factoid, multi-hop, cross-doc) và toàn bộ.
    """
    metric_map = {
        "Comprehensiveness": ("flat_comprehensiveness", "graph_comprehensiveness"),
        "Faithfulness": ("flat_faithfulness", "graph_faithfulness"),
        "Multi-hop reasoning": ("flat_multi_hop_reasoning", "graph_multi_hop_reasoning"),
        "Latency (s)": ("flat_latency_s", "graph_latency_s"),
        "Token usage": ("flat_total_tokens", "graph_total_tokens"),
    }

    rows = []
    # 1. Phân tích theo từng nhóm câu hỏi
    for group, g in eval_df.groupby("group"):
        for metric, (fc, gc) in metric_map.items():
            f = pd.to_numeric(g[fc], errors="coerce").mean()
            gr = pd.to_numeric(g[gc], errors="coerce").mean()

            if metric in {"Latency (s)", "Token usage"}:
                comment = "Flat RAG thường rẻ/nhanh hơn." if f < gr else "GraphRAG không đắt hơn trong sample này."
            else:
                delta = gr - f
                if delta >= 0.5:
                    comment = "GraphRAG cải thiện rõ; kiểm tra rationale và provenance."
                elif delta <= -0.5:
                    comment = "Flat RAG tốt hơn; graph extraction/retrieval có thể gây mất thông tin hoặc nhiễu."
                else:
                    comment = "Hai phương pháp gần nhau."

            rows.append({
                "Loại câu hỏi": group,
                "Metric": metric,
                "Flat RAG": round(f, 3) if pd.notna(f) else np.nan,
                "GraphRAG": round(gr, 3) if pd.notna(gr) else np.nan,
                "Delta": round(gr - f, 3) if (pd.notna(gr) and pd.notna(f)) else np.nan,
                "Nhận xét phân tích": comment
            })

    # 2. Tổng quan toàn bộ tập dữ liệu
    for metric, (fc, gc) in metric_map.items():
        f = pd.to_numeric(eval_df[fc], errors="coerce").mean()
        gr = pd.to_numeric(eval_df[gc], errors="coerce").mean()

        if metric in {"Latency (s)", "Token usage"}:
            comment = "Flat RAG thường rẻ/nhanh hơn." if f < gr else "GraphRAG không đắt hơn trong sample này."
        else:
            delta = gr - f
            if delta >= 0.5:
                comment = "GraphRAG cải thiện rõ trên toàn bộ tập test."
            elif delta <= -0.5:
                comment = "Flat RAG tốt hơn."
            else:
                comment = "Hai phương pháp gần nhau."

        rows.append({
            "Loại câu hỏi": "ALL (Overall)",
            "Metric": metric,
            "Flat RAG": round(f, 3) if pd.notna(f) else np.nan,
            "GraphRAG": round(gr, 3) if pd.notna(gr) else np.nan,
            "Delta": round(gr - f, 3) if (pd.notna(gr) and pd.notna(f)) else np.nan,
            "Nhận xét phân tích": comment
        })

    return pd.DataFrame(rows)
