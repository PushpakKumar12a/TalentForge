import json
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer

def extract_necessary_features(candidate):

    KEYWORDS = {
    # Embeddings / Retrieval
    "sentence-transformer", "sentence-transformers", "openai embedding", "bge", "e5",
    "embedding", "semantic search", "vector search",
    "retrieval", "ranking", "hybrid search", "hybrid retrieval", "bm25", "re-ranking",

    # Vector DBs
    "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss",
    "vector database",

    # Evaluation
    "ndcg", "mrr", "map", "evaluation",
    "a/b test", "offline-to-online correlation",

    # Core Languages & LLM fine tuning
    "python", "lora", "qlora", "peft",
    "fine-tuning", "llm",

    # Ranking
    "learning-to-rank", "ltr",
    "xgboost", "neural",

    # Domain experience
    "hr-tech", "recruiting tech",
    "marketplace",

    # Systems
    "distributed systems",
    "large-scale inference",
    "inference optimization",

    # Open source
    "open source",
    "open-source",
    "github contributor"
    }

    BONUS_SIGNALS = {
    # Fine-tuning & ltr
    "lora": 5,
    "qlora": 5,
    "peft": 5,
    "xgboost": 4,
    "neural": 4,
    "learning-to-rank": 6,
    
    # Domain & system experience
    "hr-tech": 4,
    "recruiting tech": 4,
    "marketplace": 3,
    "distributed systems": 4,
    "large-scale inference": 4,
    "inference optimization": 4,
    "open source": 3
    }

    NEGATIVE_SIGNALS = {
    # Explicit disqualifiers from JD
    "academic labs": -10,
    "pure research": -10,
    "research-only": -10,
    "closed-source": -5,
    
    # Tool/Framework caveats
    "langchain": -2,
    
    # Domain mismatches
    "computer vision": -5,
    "opencv": -4,
    "speech recognition": -5,
    "speech": -5,
    "robotics": -5,
    
    # Pure consulting limitations
    "consulting": -3,
    "tcs": -3,
    "infosys": -3,
    "wipro": -3,
    "accenture": -3,
    "cognizant": -3,
    "capgemini": -3
    }

    features = []
    
    # 1. Base profile data
    profile = candidate.get("profile", {})
    features.append(f"Title: {profile.get('current_title', '')}")
    features.append(f"Exp: {profile.get('years_of_experience', 0)} years")
    features.append(f"Industry: {profile.get('current_industry', '')}")
    features.append(f"Location: {profile.get('location', '')}, {profile.get('country', '')}")
    
    # 2. Behavioral & Availability
    signals = candidate.get("redrob_signals", {})
    features.append(f"Last Active: {signals.get('last_active_date', '')}")
    features.append(f"Recruiter Response Rate: {signals.get('recruiter_response_rate', 0)}")
    features.append(f"Willing to Relocate: {signals.get('willing_to_relocate', False)}")
    
    # 3. Filtered Skills
    matched_skills = []
    for skill in candidate.get("skills", []):
        s = skill.get("name", "").lower()
        if any(k in s for k in KEYWORDS):
            matched_skills.append(s)
    
    if matched_skills:
        features.append("Relevant Skills: " + ", ".join(matched_skills))
        
    # 4. Filtered Career Details
    career_notes = []
    for job in candidate.get("career_history", []):
        desc = (job.get("description") or "").lower()
        title = (job.get("title") or "").lower()
        industry = (job.get("industry") or "").lower()
        company = job.get("company") or job.get("company_name") or ""
        
        is_relevant = any(k in desc or k in title for k in KEYWORDS) or "ai" in title.split() or "ml" in title.split() or "machine learning" in title or "recommendation" in desc or "search" in desc or "ranking" in desc
        
        is_service = "services" in industry or "consulting" in industry or "outsourcing" in industry
        comp_type = "Product/Domain Company" if not is_service and industry != "" else "Service/Consulting Company"
        
        if is_relevant:
            career_notes.append(f"{job.get('title', '')} at {company} [{comp_type}]: {job.get('description', '')[:200]}")
            
    if career_notes:
        features.append("Relevant Experience: " + " | ".join(career_notes))
    return " ".join(features)

def main():
    print("Loading candidate data...")
    cands_texts = []
    cands_ids = []
    
    with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            cands_ids.append(c["candidate_id"])
            cands_texts.append(extract_necessary_features(c))
            
    print(f"Extracted focused features for {len(cands_texts)} candidates.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer on {device.upper()}...")
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "bge-small-en-v1.5")
    model = SentenceTransformer(model_path, device=device)

    print("Generating dense embeddings...")
    batch_size = 256
    
    embeddings = model.encode(
        cands_texts, 
        batch_size=batch_size, 
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    print("Saving precomputed features to disk...")
    np.save("data/candidate_embeddings.npy", embeddings)
    with open("data/candidate_ids.json", "w") as f:
        json.dump(cands_ids, f)
        
    print("Precomputation complete! Output securely cached at data/candidate_embeddings.npy")

if __name__ == "__main__":
    main()
