import argparse, csv, gc, json, os, resource, sys, time
from pathlib import Path
import bm25s
import Stemmer
import numpy as np
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer
from pipeline.config import (
    JD_BM25, JD_SEMANTIC, BM25_TOP_K, FINAL_TOP_K,
    FLASHRANK_BATCH_SIZE, PROJECT_DIR,
)
from pipeline.filters import is_honeypot, passes_prefilter
from pipeline.text_builder import build_text_short, build_text_full
from pipeline.scoring import behavioral_multiplier, rank_based_score
from pipeline.reasoning import generate_reasoning

# Memory + thread limits
MAX_RAM = 16 * 1024 * 1024 * 1024  # 16 GiB
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM, hard))
except (ValueError, resource.error):
    pass

# Auto assign threads based on available RAM
try:
    total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    if total_ram_gb > 8:
        num_threads = "3"
    else:
        num_threads = "2"
except Exception:
    num_threads = "2"

os.environ.setdefault("OMP_NUM_THREADS", num_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", num_threads)
os.environ.setdefault("MKL_NUM_THREADS", num_threads)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
print(f"System RAM: {total_ram_gb:.1f}GB | Using {num_threads} threads")


def mem_mb():
    # Current process RSS in MB (Linux).
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Redrob candidate ranking pipeline")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()
    cpath, opath = Path(args.candidates), Path(args.out)

    # Stage 1: Stream + filter
    print("[1/5] Streaming + filtering …")
    texts, cand_ids, offsets = [], [], []
    total = hp = filt = 0

    with open(cpath, "rb") as fh:
        while True:
            off = fh.tell()
            line = fh.readline()
            if not line:
                break
            total += 1
            c = json.loads(line)
            if is_honeypot(c):
                hp += 1; del c; continue
            if not passes_prefilter(c):
                filt += 1; del c; continue
            texts.append(build_text_short(c))
            cand_ids.append(c["candidate_id"])
            offsets.append(off)
            del c

    kept = len(texts)
    print(f"    Total={total}  Honeypots={hp}  Filtered={filt}  Kept={kept}")
    print(f"    Stage 1: {time.time()-t0:.1f}s  RSS={mem_mb():.0f}MB")
    gc.collect()

    # Stage 2: Hybrid Retrieval (BM25 + Dense Vectors)
    t1 = time.time()
    print("[2/5] Hybrid indexing + retrieval (BM25 + Embeddings) …")
    
    # 2a. BM25
    stemmer = Stemmer.Stemmer("english")
    corpus_tok = bm25s.tokenize(texts, stemmer=stemmer, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tok)

    q_tok = bm25s.tokenize([JD_BM25], stemmer=stemmer, stopwords="en")
    k = min(BM25_TOP_K, kept)
    
    # Retrieve all valid candidates to combine scores
    res, bm25_scores_full = retriever.retrieve(q_tok, k=kept)
    bm25_sorted_idxs = res[0]
    bm25_sorted_scores = bm25_scores_full[0]
    
    bm25_original_order = np.zeros(kept, dtype=np.float32)
    bm25_original_order[bm25_sorted_idxs] = bm25_sorted_scores
    
    del retriever, corpus_tok, q_tok, texts, stemmer, res, bm25_scores_full
    gc.collect()

    # 2b. Dense Embeddings
    print("    Loading embeddings...")
    all_dense_embs = np.load("data/candidate_embeddings.npy")
    with open("data/candidate_ids.json", "r") as f:
        all_dense_ids = json.load(f)
        
    dense_id_to_idx = {cid: i for i, cid in enumerate(all_dense_ids)}
    
    valid_dense_embs = np.zeros((kept, all_dense_embs.shape[1]), dtype=np.float32)
    for i, cid in enumerate(cand_ids):
        valid_dense_embs[i] = all_dense_embs[dense_id_to_idx[cid]]
        
    del all_dense_embs, all_dense_ids, dense_id_to_idx
    gc.collect()
    
    print("    Embedding JD and computing cosine similarities...")
    model = SentenceTransformer(str(PROJECT_DIR / "weights" / "bge-small-en-v1.5"), device="cpu")
    jd_emb = model.encode([JD_SEMANTIC], normalize_embeddings=True)[0]
    dense_scores = valid_dense_embs @ jd_emb
    
    del model, valid_dense_embs
    gc.collect()

    # 2c. Score Normalization & Fusion
    def normalize(arr):
        min_v, max_v = arr.min(), arr.max()
        if max_v > min_v:
            return (arr - min_v) / (max_v - min_v)
        return arr
        
    dense_norm = normalize(dense_scores)
    bm25_norm = normalize(bm25_original_order)
    
    # 70% Dense Semantic, 30% Exact BM25 Keyword match
    hybrid_scores = 0.7 * dense_norm + 0.3 * bm25_norm
    
    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    bm25_idxs = top_indices.tolist()

    print(f"    Shortlisted {len(bm25_idxs)} candidates via Hybrid Search")
    print(f"    Stage 2: {time.time()-t1:.1f}s  RSS={mem_mb():.0f}MB")

    # Stage 3: Re-read shortlisted candidates
    t2 = time.time()
    print("[3/5] Re-reading shortlisted candidates …")
    need = sorted({offsets[i] for i in bm25_idxs})
    off2cand = {}
    with open(cpath, "rb") as fh:
        for off in need:
            fh.seek(off)
            off2cand[off] = json.loads(fh.readline())

    rerank_cands, rerank_texts = [], []
    for i in bm25_idxs:
        c = off2cand[offsets[i]]
        rerank_cands.append(c)
        rerank_texts.append(build_text_full(c))

    del off2cand, need, bm25_idxs, offsets, cand_ids
    gc.collect()
    print(f"    Loaded {len(rerank_cands)} candidates")
    print(f"    Stage 3: {time.time()-t2:.1f}s  RSS={mem_mb():.0f}MB")

    # Stage 4: FlashRank batch reranking
    t3 = time.time()
    BS = FLASHRANK_BATCH_SIZE
    n = len(rerank_texts)
    n_batches = (n + BS - 1) // BS
    print(f"[4/5] FlashRank reranking ({BS}/batch, {n_batches} batches) …")

    ranker = Ranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        cache_dir=str(PROJECT_DIR / "weights"),
    )

    all_results = []
    for b in range(n_batches):
        start, end = b * BS, min((b + 1) * BS, n)
        passages = [{"id": start + j, "text": rerank_texts[start + j]}
                    for j in range(end - start)]
        rr = ranker.rerank(RerankRequest(query=JD_SEMANTIC, passages=passages))
        all_results.extend(rr)
        del passages, rr; gc.collect()
        print(f"      Batch {b+1}/{n_batches} done  RSS={mem_mb():.0f}MB")

    del ranker, rerank_texts
    gc.collect()
    print(f"    Reranked {len(all_results)} passages")
    print(f"    Stage 4: {time.time()-t3:.1f}s  RSS={mem_mb():.0f}MB")

    # Stage 5: Score + output
    t4 = time.time()
    print("[5/5] Scoring + output …")

    scored = []
    for item in all_results:
        pid = item.get("id")
        if pid is None or int(pid) >= len(rerank_cands):
            continue
        c = rerank_cands[int(pid)]
        final = float(item.get("score", 0)) * behavioral_multiplier(c)
        scored.append((final, c["candidate_id"], c))

    del all_results; gc.collect()

    # Deterministic sort: score DESC, candidate_id ASC
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:FINAL_TOP_K]

    rows = []
    with open(opath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        
        # Write each row to disk as reasoning is generated
        for rank, (_, cid, c) in enumerate(top, 1):
            row = {
                "candidate_id": cid,
                "rank": rank,
                "score": f"{rank_based_score(rank):.6f}",
                "reasoning": generate_reasoning(c, rank),
            }
            w.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"\r      -> Generated reasoning for rank {rank}/{len(top)}", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n Streamed and wrote {len(rows)} rows to {opath} in {elapsed:.1f}s")

    # Sanity checks
    assert len(rows) <= FINAL_TOP_K
    sc = [float(r["score"]) for r in rows]
    for i in range(1, len(sc)):
        assert sc[i] <= sc[i - 1], f"Non-monotonic at rank {i + 1}"
    print("  All checks passed.")

if __name__ == "__main__":
    main()
