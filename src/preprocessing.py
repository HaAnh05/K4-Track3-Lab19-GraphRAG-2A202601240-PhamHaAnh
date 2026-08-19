import re
import json
import hashlib
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from src.config import (
    HF_TOKEN,
    SEED,
    CHUNK_WORDS,
    CHUNK_OVERLAP_WORDS,
    LAB_MAX_ARTICLES,
    LAB_MAX_CHUNKS,
    EXTRACTION_MAX_CHUNKS,
    OUTPUT_DIR,
)
from src.llm_client import groq_json

def norm_space(x) -> str:
    """Chuẩn hóa khoảng trắng thừa."""
    return re.sub(r"\s+", " ", str(x or "")).strip()

def sha1(x) -> str:
    """Tạo mã băm SHA-1 cho chuỗi dữ liệu."""
    return hashlib.sha1(str(x).encode("utf-8", errors="ignore")).hexdigest()

def pick_col(df: pd.DataFrame, candidates: list, required: bool = True):
    """Tìm tên cột tương ứng trong DataFrame không phân biệt hoa thường."""
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise KeyError(f"Missing one of columns: {candidates}")
    return None

def stream_hackernoon_to_csv(output_path: str, target_mb: int = 300, max_rows: int = 50000, prioritize_mb: bool = True) -> pd.DataFrame:
    """
    Stream dữ liệu tin tức công nghệ từ Hugging Face và lưu ra CSV cục bộ.
    """
    from datasets import load_dataset
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Connecting HF stream: HackerNoon/tech-company-news-data-dump ...")
    ds = load_dataset(
        "HackerNoon/tech-company-news-data-dump",
        split="train",
        streaming=True,
        token=HF_TOKEN or None,
    )

    rows = []
    total_bytes = 0
    target_bytes = target_mb * 1024 * 1024

    for i, item in enumerate(tqdm(ds, desc="Streaming HackerNoon")):
        title = str(item.get("title") or "")
        text = str(item.get("text") or "")
        row = {
            "title": title,
            "text": text,
            "published_date": str(item.get("published_date") or item.get("date") or ""),
            "id": str(item.get("id") or item.get("article_id") or i),
        }
        rows.append(row)
        total_bytes += len(title.encode("utf-8")) + len(text.encode("utf-8"))

        if (i + 1) >= max_rows:
            print(f"Reached max_rows={max_rows}.")
            break
        if prioritize_mb and total_bytes >= target_bytes:
            print(f"Reached ~{total_bytes / (1024*1024):.1f} MB.")
            break

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} articles ({out.stat().st_size / (1024*1024):.1f} MB) -> {out}")
    return df

def load_news(path: str) -> pd.DataFrame:
    """Nạp file dữ liệu tin tức với các định dạng phổ biến."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    raise ValueError(f"Định dạng không được hỗ trợ: {suffix}")

def standardize_news(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa dữ liệu tin tức và khử trùng lặp chính xác bằng SHA-1.
    """
    text_col = pick_col(raw, ["text", "content", "article", "body", "story", "description"])
    title_col = pick_col(raw, ["title", "headline"], required=False)
    date_col = pick_col(raw, ["published_date", "date", "published_at", "created_at"], required=False)
    id_col = pick_col(raw, ["id", "article_id", "story_id", "uuid"], required=False)

    df = pd.DataFrame()
    df["text"] = raw[text_col].fillna("").map(norm_space)
    df["title"] = raw[title_col].fillna("").map(norm_space) if title_col else ""

    if date_col:
        df["published_date"] = (
            pd.to_datetime(raw[date_col], errors="coerce", utc=True)
            .dt.strftime("%Y-%m-%d")
            .fillna("")
        )
    else:
        df["published_date"] = ""

    if id_col:
        df["article_id"] = raw[id_col].astype(str)
    else:
        df["article_id"] = [
            sha1(f"{t}\n{x}")[:20] for t, x in zip(df["title"], df["text"])
        ]

    # Lọc bài viết ngắn
    df = df[df["text"].str.len() >= 80].copy()
    df["dedup_key"] = [
        sha1(norm_space(f"{t}\n{x}").lower())
        for t, x in zip(df["title"], df["text"])
    ]
    before = len(df)
    df = df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)
    print(f"Exact dedup: {before:,} -> {len(df):,}")

    if LAB_MAX_ARTICLES and len(df) > LAB_MAX_ARTICLES:
        df = df.sample(LAB_MAX_ARTICLES, random_state=SEED).sort_index().reset_index(drop=True)
    return df

def chunk_text(text: str, size: int = 220, overlap: int = 40) -> list:
    """
    Cắt text thành các chunk theo rolling window có overlap.
    """
    words = norm_space(text).split()
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        part = words[start:start+size]
        if not part:
            break
        out.append(" ".join(part))
        if start + size >= len(words):
            break
    return out

def build_chunks(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Xây dựng bảng chunks từ DataFrame bài viết.
    """
    rows = []
    for r in tqdm(news_df.itertuples(index=False), total=len(news_df), desc="Chunking"):
        for i, text in enumerate(chunk_text(r.text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)):
            rows.append({
                "chunk_id": f"{r.article_id}::c{i:04d}",
                "article_id": r.article_id,
                "title": r.title,
                "published_date": r.published_date,
                "text": text,
            })
            if LAB_MAX_CHUNKS and len(rows) >= LAB_MAX_CHUNKS:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)

COREF_SYSTEM = """
You are a conservative coreference-resolution component for a knowledge-graph pipeline.
Resolve pronouns and generic references only when the antecedent is clearly supported in the same chunk.
Never invent facts. Preserve dates, numbers, tickers and product names.
Return strict JSON only.
""".strip()

def resolve_coref_batch(batch_df: pd.DataFrame):
    """
    Phân giải đại từ theo từng batch nhỏ thông qua LLM.
    """
    payload = [
        {"chunk_id": r.chunk_id, "text": r.text}
        for r in batch_df.itertuples(index=False)
    ]

    prompt = f"""
Resolve coreferences.

Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "resolved_text": "...",
      "unresolved_mentions": ["..."]
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    obj, usage = groq_json(COREF_SYSTEM, prompt)
    by_id = {x.get("chunk_id"): x for x in obj.get("items", [])}

    rows = []
    for r in batch_df.itertuples(index=False):
        item = by_id.get(r.chunk_id, {})
        rows.append({
            "chunk_id": r.chunk_id,
            "resolved_text": norm_space(item.get("resolved_text") or r.text),
            "unresolved_mentions": item.get("unresolved_mentions", []),
        })
    return pd.DataFrame(rows), usage

def run_coref(chunks_subset: pd.DataFrame, batch_size: int = 5) -> pd.DataFrame:
    """
    Thực thi Coreference Resolution cho toàn bộ tập chunk cần trích xuất đồ thị.
    """
    out = []
    for start in tqdm(range(0, len(chunks_subset), batch_size), desc="Coref"):
        batch = chunks_subset.iloc[start:start+batch_size]
        try:
            df, _ = resolve_coref_batch(batch)
        except Exception as e:
            print(f"[Coref Warning] Batch {start} failed: {e}. Fallback text gốc.")
            df = pd.DataFrame({
                "chunk_id": batch["chunk_id"].tolist(),
                "resolved_text": batch["text"].tolist(),
                "unresolved_mentions": [["COREF_BATCH_FAILED"] for _ in range(len(batch))],
            })
        out.append(df)
    return pd.concat(out, ignore_index=True)
