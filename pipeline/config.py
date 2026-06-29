from datetime import date
from pathlib import Path

# Runtime
TODAY = date(2026, 6, 17)
INACTIVE_CUTOFF_DAYS = 180
MIN_RESPONSE_RATE = 0.10
BM25_TOP_K = 500
FINAL_TOP_K = 100
FLASHRANK_BATCH_SIZE = 128
PROJECT_DIR = Path(__file__).resolve().parent.parent
MAX_TEXT_BM25 = 600
MAX_TEXT_RERANK = 1024

# JD queries
JD_BM25 = (
    "Senior AI Engineer machine learning embeddings retrieval ranking "
    "LLM fine-tuning production Python vector database search "
    "sentence-transformers FAISS Pinecone Weaviate Qdrant Elasticsearch "
    "hybrid search semantic search NLP information retrieval recommendation "
    "NDCG MRR MAP deep learning PyTorch TensorFlow transformer BERT "
    "applied ML product company startup LoRA QLoRA PEFT MLOps "
    "ranking system learning to rank XGBoost FastAPI Docker Kubernetes"
)
JD_SEMANTIC = (
    "Senior AI Engineer for founding team at Redrob AI, Series A "
    "talent intelligence platform in Pune/Noida India. 5-9 years "
    "experience in embeddings, retrieval, ranking, LLMs. Must have "
    "production embeddings-based retrieval (sentence-transformers, BGE, E5), "
    "vector databases (Pinecone, Weaviate, Qdrant, FAISS, Elasticsearch), "
    "strong Python, evaluation frameworks (NDCG, MRR, MAP). "
    "Nice-to-have: LLM fine-tuning, learning-to-rank, HR-tech. "
    "Not suitable: pure consulting, CV/speech/robotics only, no production."
)

# Lookup data
CORE_SKILLS = {
    "sentence-transformers", "embeddings", "semantic search", "vector search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "opensearch", "hybrid search", "information retrieval", "vector database",
    "nlp", "natural language processing", "transformers", "bert", "gpt",
    "machine learning", "deep learning", "neural networks",
    "pytorch", "tensorflow", "scikit-learn",
    "llm", "large language models", "fine-tuning", "fine-tuning llms",
    "lora", "qlora", "peft", "prompt engineering", "rag",
    "retrieval augmented generation",
    "search", "ranking", "recommendation systems", "learning to rank",
    "bm25", "tf-idf", "recommendation engine",
    "python", "rest api", "fastapi", "flask", "docker", "kubernetes",
    "mlops", "aws", "gcp", "azure", "sql", "spark", "airflow",
}
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mindtree", "mphasis", "ltimindtree", "cts",
}
LOC_PRI = {"pune", "noida"}
LOC_SEC = {
    "hyderabad", "mumbai", "delhi", "delhi ncr", "bengaluru",
    "bangalore", "gurgaon", "gurugram", "new delhi", "chennai",
}
TITLE_TIERS = {
    "strong":   (["ai engineer", "ml engineer", "machine learning engineer",
                  "nlp engineer", "search engineer", "ranking engineer",
                  "applied scientist", "research engineer", "deep learning engineer",
                  "senior ai", "staff ai", "lead ai", "senior ml", "staff ml", "lead ml"], 1.15),
    "moderate": (["data scientist", "software engineer", "backend engineer",
                  "full stack", "platform engineer", "senior software",
                  "staff engineer", "tech lead", "data engineer", "analytics engineer"], 1.02),
    "red_flag": (["hr manager", "accountant", "sales executive", "marketing manager",
                  "content writer", "graphic designer", "operations manager",
                  "project manager", "customer support", "business analyst",
                  "civil engineer", "mechanical engineer", "electrical engineer"], 0.50),
}
TITLE_DEFAULT_MULT = 0.85
