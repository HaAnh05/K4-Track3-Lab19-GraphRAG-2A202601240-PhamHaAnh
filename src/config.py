import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Nạp file .env nếu có
load_dotenv()

# Thiết lập thư mục outputs
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.max_colwidth", 120)

def get_secret(name: str, default=None):
    """
    Lấy giá trị cấu hình từ biến môi trường hoặc Google Colab userdata.
    """
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value is not None:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)

# Neo4j configuration (hỗ trợ cả NEO4J_USER và NEO4J_USERNAME)
NEO4J_URI = get_secret("NEO4J_URI", "")
NEO4J_USER = get_secret("NEO4J_USER", get_secret("NEO4J_USERNAME", "neo4j"))
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "")
NEO4J_DATABASE = get_secret("NEO4J_DATABASE", "neo4j")

# LLM configuration (Groq & OpenAI) - Sử dụng openai/gpt-oss-20b mặc định tốc độ cao & ổn định
GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")
if GROQ_MODEL == "openai/gpt-oss-120b":
    GROQ_MODEL = "openai/gpt-oss-20b"

JUDGE_PROVIDER = get_secret("JUDGE_PROVIDER", "groq").lower()
JUDGE_MODEL = get_secret("JUDGE_MODEL", "openai/gpt-oss-20b")
if JUDGE_MODEL == "openai/gpt-oss-120b":
    JUDGE_MODEL = "openai/gpt-oss-20b"

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
HF_TOKEN = get_secret("HF_TOKEN", "")

# Data paths
if Path("data/graphrag_golden_50_first5000.csv").exists():
    GOLDEN_PATH = "data/graphrag_golden_50_first5000.csv"
elif Path("data/golden_dataset.csv").exists():
    GOLDEN_PATH = "data/golden_dataset.csv"
elif Path("/content/data/graphrag_golden_50_first5000.csv").exists():
    GOLDEN_PATH = "/content/data/graphrag_golden_50_first5000.csv"
elif Path("/content/golden_dataset.csv").exists():
    GOLDEN_PATH = "/content/golden_dataset.csv"
else:
    GOLDEN_PATH = "data/golden_dataset.csv"

if Path("data/hackernoon_subset.csv").exists():
    DATA_PATH = "data/hackernoon_subset.csv"
elif Path("/content/data/hackernoon_subset.csv").exists():
    DATA_PATH = "/content/data/hackernoon_subset.csv"
elif Path("/content/hackernoon_subset.csv").exists():
    DATA_PATH = "/content/hackernoon_subset.csv"
else:
    DATA_PATH = "data/hackernoon_subset.csv"

# Pipeline Scale Guard limits
LAB_MAX_ARTICLES = 1500
LAB_MAX_CHUNKS = 3000
EXTRACTION_MAX_CHUNKS = 400
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

# Graph Schema Allowlist
ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED", "DEVELOPED", "INVESTED_IN", "FOUNDED",
    "WORKED_AT", "PARTNERED_WITH", "USES", "LEADS"
}

# Graph Traversal & Super-node Mitigation Constants
SUPER_NODE_DEGREE = 100
SUPER_NODE_EDGE_CAP = 50
GLOBAL_EDGE_CAP = 250
MAX_GRAPH_CONTEXT_CHARS = 14000

# Embedding model singleton
_embedder = None

def get_embedder():
    """
    Singleton instance của SentenceTransformer embedding model.
    """
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder
