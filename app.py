import streamlit as st
import time
import pandas as pd
import subprocess
import os

st.set_page_config(
    page_title="Candidate Ranking Pipeline",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.st-emotion-cache-b0dv7m {
    background-color: #FF4B4B;
}
</style>
""", unsafe_allow_html=True)

if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'stopped' not in st.session_state:
    st.session_state.stopped = False
if 'finished' not in st.session_state:
    st.session_state.finished = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0.0
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0.0

def run_pipeline():
    st.session_state.is_running = True
    st.session_state.stopped = False
    st.session_state.finished = False
    st.session_state.start_time = time.time()
    st.session_state.elapsed_time = 0.0

def stop_pipeline():
    st.session_state.is_running = False
    st.session_state.stopped = True
    st.session_state.finished = False
    if st.session_state.start_time > 0:
        st.session_state.elapsed_time = time.time() - st.session_state.start_time

col_title, col_github = st.columns([10, 2])
with col_title:
    st.title("🎯 Candidate Ranking Pipeline")
with col_github:
    st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)
    st.link_button("🐙 View on GitHub", url="https://github.com/PushpakKumar12a/Candidate-Ranking", width='stretch')

col_left, col_right = st.columns([1.1, 2])

with col_left:
    with st.container(border=True):
        uploaded_file = st.file_uploader("File Uploading section")
    with st.container(border=True):
        use_sample = st.checkbox("Candidate Sample Data", value=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    timer_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.is_running:
        with timer_placeholder:
            st.iframe("""
                <style>
                        body { margin: 0; padding: 0 }
                        #timer {font-family: sans-serif; font-size: 16px; font-weight: 600; }
                </style>
                <div id="timer">Elapsed Time: 0.0s</div>
                <script>
                    const timerEl = document.getElementById('timer');
                    function syncTheme() {
                        try {
                            const parentColor = window.parent.getComputedStyle(window.parent.document.body).color;
                            if (parentColor) timerEl.style.color = parentColor;
                        } catch(e) {}
                    }
                    syncTheme();
                    setInterval(syncTheme, 1000);

                    const start = Date.now();
                    setInterval(() => {
                        const elapsed = ((Date.now() - start) / 1000).toFixed(1);
                        if (timerEl) {
                            timerEl.innerText = `Elapsed Time: ${elapsed}s`;
                        }
                    }, 100);
                </script>
            """,height=30)
    elif st.session_state.elapsed_time > 0:
        timer_placeholder.markdown(f'<div style="font-size:16px; font-weight:600; margin-bottom: 15px;">Elapsed Time: {st.session_state.elapsed_time:.1f}s</div>', unsafe_allow_html=True)
    else:
        timer_placeholder.markdown('<div style="font-size:16px; font-weight:600; margin-bottom: 15px;">Elapsed Time: 0.0s</div>', unsafe_allow_html=True)
    
    btn_col1, btn_col2 = st.columns(2)
    
    can_run = (uploaded_file is not None or use_sample) and not st.session_state.is_running
    
    with btn_col1:
        st.button("Run Pipeline", type="primary", width='stretch', 
                  disabled=not can_run, on_click=run_pipeline)
    with btn_col2:
        st.button("Stop Pipeline", type="secondary", width='stretch', 
                  disabled=not st.session_state.is_running, on_click=stop_pipeline)

st.markdown("""
<style>
.info-container {
    height: 390px; 
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

with col_right:
    with st.container(border=True):
        st.subheader("⚙️ About the Pipeline flow")
        st.markdown("""
        > **This pipeline evaluates and ranks candidate profiles using a state-of-the-art, highly optimized 5-stage architecture designed for scale and precision:**

        1. **Streaming & Filtering:** Ingests large-scale `JSONL` candidate data via memory-efficient streaming. Applies fast heuristic filters to instantly discard incomplete or unqualified profiles, drastically reducing memory overhead.
        2. **Hybrid Retrieval:** Leverages a dual-encoder architecture. It combines sparse `BM25` lexical matching (for exact keyword extraction) with dense vector embeddings (e.g., `MiniLM`) to capture semantic meaning and contextual relevance, yielding a highly accurate initial shortlist.
        3. **Candidate Profiling:** Performs a secondary pass to re-read the complete, rich data profiles exclusively for the shortlisted candidates. This isolates memory usage and prepares the deep textual features required for complex downstream analysis.
        4. **Batch Reranking:** Applies advanced cross-encoder models via `FlashRank`. By executing the reranking in dynamically optimized batches, it computes highly precise query-document relevance scores without exhausting system resources.
        5. **Scoring & Export:** Aggregates and normalizes the final compatibility scores, ranks the candidates in descending order of relevance, and generates a structured output CSV ready for immediate integration or human review.
        """)

st.markdown("<br>", unsafe_allow_html=True)

col_config = {
    "candidate_id": st.column_config.TextColumn("candidate_id", width="small"),
    "rank": st.column_config.NumberColumn("rank", width="small"),
    "score": st.column_config.NumberColumn("score", width="small"),
    "reasoning": st.column_config.TextColumn("reasoning", width="large")
}

preview_container = st.container(border=True)
with preview_container:
    st.subheader("📊 CSV Preview Section")
    df_placeholder = st.empty()
    if not st.session_state.finished:
        empty_df = pd.DataFrame(columns=["candidate_id", "rank", "score", "reasoning"])
        df_placeholder.dataframe(empty_df, width='stretch', hide_index=True, column_config=col_config)

if st.session_state.stopped:
    status_placeholder.warning("Pipeline was stopped.")
    st.session_state.stopped = False 
    
if st.session_state.is_running:
    progress_bar = progress_placeholder.progress(0)
    
    if uploaded_file is not None:
        cands_path = "uploaded_candidates.jsonl"
        with open(cands_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    else:
        cands_path = "data/sample_candidates.jsonl"
        
    os.makedirs("generated", exist_ok=True)
    out_csv = "generated/submission_nanopixel.csv"
    
    process = subprocess.Popen(
        ["python", "rank.py", "--candidates", cands_path, "--out", out_csv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    try:
        for line in iter(process.stdout.readline, ''):
            if not line:
                continue
            
            if "[1/5]" in line:
                status_placeholder.text("Stage 1/5: Streaming and filtering candidates...")
                progress_bar.progress(10)
            elif "[2/5]" in line:
                status_placeholder.text("Stage 2/5: Hybrid Retrieval (BM25 + Dense Vectors)...")
                progress_bar.progress(15)
            elif "[3/5]" in line:
                status_placeholder.text("Stage 3/5: Re-reading shortlisted candidates...")
                progress_bar.progress(45)
            elif "[4/5]" in line:
                status_placeholder.text("Stage 4/5: FlashRank batch reranking...")
                progress_bar.progress(80)
            elif "[5/5]" in line:
                status_placeholder.text("Stage 5/5: LLM reasoning & scoring...")
                progress_bar.progress(85)
            elif "Streamed and wrote" in line:
                progress_bar.progress(100)
                
        process.wait()
        
        if process.returncode == 0:
            st.session_state.is_running = False
            st.session_state.finished = True
            st.session_state.elapsed_time = time.time() - st.session_state.start_time
            st.rerun()
        else:
            st.session_state.is_running = False
            st.session_state.elapsed_time = time.time() - st.session_state.start_time
            status_placeholder.error("Pipeline failed! Check terminal logs.")
            st.rerun()
            
    except BaseException as e:
        if process.poll() is None:
            process.kill()
        raise e

if st.session_state.finished:
    status_placeholder.success("Pipeline completed successfully!")
    progress_placeholder.progress(100)
    
    out_csv = "generated/submission_nanopixel.csv"
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        
        with preview_container:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results",
                data=csv_data,
                file_name="submission_nanopixel.csv",
                mime="text/csv",
                type="primary"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            df_placeholder.dataframe(
                df, 
                width='stretch', 
                hide_index=True,
                column_config=col_config
            )
    else:
        with preview_container:
            df_placeholder.error("Output CSV not found.")