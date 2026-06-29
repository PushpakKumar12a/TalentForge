import random
import re
import os

# Pre defined core skills
AI_KEYWORDS = {
    "llm", "rag", "pytorch", "embeddings", "pinecone", "fine-tuning", 
    "machine learning", "nlp", "genai", "python", "fastapi", "docker",
    "tensorflow", "keras", "scikit-learn", "lstm", "naive bayes", 
    "computer vision", "cnn", "transformers", "langchain", "llamaindex", 
    "huggingface", "prompt engineering", "faiss", "milvus", "weaviate", 
    "pandas", "numpy", "sql", "aws", "kubernetes", "docker compose", 
    "mlflow", "airflow", "django", "flask", "node.js", "express", 
    "react", "rest api", "qlora", "diffusion models", "data science",
    "deep learning", "predictive modeling", "sentiment analysis"
}

# LLM Singleton
llm = None

def get_llm():
    global llm
    if llm is None:
        from llama_cpp import Llama
        from .config import PROJECT_DIR
        
        llm = Llama(
            model_path=str(PROJECT_DIR / "weights" / "llama-3.2-1b-instruct-q4_k_m.gguf"),
            n_ctx=512,
            n_threads=int(os.environ.get("OMP_NUM_THREADS", "4")), 
            n_batch=256,
            verbose=False
        )
    return llm

# Data Extraction & HR logic
def extract_candidate_facts(candidate):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    title = profile.get("current_title", "Tech Professional")
    exp = profile.get("years_of_experience", 0)
    
    skills_list = [s.get("name") for s in candidate.get("skills", [])]
    core_skills = [s for s in skills_list if s.lower() in AI_KEYWORDS][:3]
    if not core_skills:
        core_skills = skills_list[:2]
    skills_str = ", ".join(core_skills) if core_skills else "general tools"
    
    notice = signals.get("notice_period_days", 30)
    industry = profile.get("current_industry", "general tech")
    
    is_immediate = "eligible for immediate hiring" if notice <= 15 else "not eligible for immediate start"
    
    if exp < 5:
        exp_gap = "lacks the required senior-level experience"
    elif exp > 10:
        exp_gap = "may be over-experienced for a founding team pace"
    elif not core_skills:
        exp_gap = "lacks direct hands-on exposure to core AI tech"
    else:
        exp_gap = "shows appropriate tenure"
        
    return title, exp, skills_str, notice, is_immediate, exp_gap, industry

# 10-tier tone generator
def get_tone_instruction(rank):

    if rank <= 10:
        return "Tone Profile (Ranks 1-10): Exceptionally positive. Frame the candidate as an elite, top-tier fit with deep alignment to complex AI systems."
    elif rank <= 20:
        return "Tone Profile (Ranks 11-20): Very strong. Highlight robust technical depth and solid foundational fit, noting them as highly capable."
    elif rank <= 30:
        return "Tone Profile (Ranks 21-30): Solid and reliable. Focus on their dependable experience and good general alignment."
    elif rank <= 40:
        return "Tone Profile (Ranks 31-40): Competent but standard. Note decent baseline skills, but hint at a lack of exceptional standout traits."
    elif rank <= 50:
        return "Tone Profile (Ranks 41-50): Borderline/Average. Frame the experience as acceptable but somewhat generic or overly broad."
    elif rank <= 60:
        return "Tone Profile (Ranks 51-60): Mediocre. Subtly emphasize noticeable gaps in scale, specific production alignment, or trajectory."
    elif rank <= 70:
        return "Tone Profile (Ranks 61-70): Weak fit. Highlight misalignment in industry, tooling focus, or required tenure."
    elif rank <= 80:
        return "Tone Profile (Ranks 71-80): Poor fit. Express clear concern over their tech stack relevance, background, or behavioral signals."
    elif rank <= 90:
        return "Tone Profile (Ranks 81-90): Highly questionable. Frame as a distinct mismatch for this fast-paced startup role, noting significant flaws."
    else:
        return "Tone Profile (Ranks 91-100): Complete mismatch. Explicitly highlight fatal flaws, irrelevant backgrounds (e.g., non-engineering), or disqualifying traits."

# Dynamic prompt builder
def get_hr_prompt(rank, title, exp, skills, notice, is_immediate, exp_gap, industry):
    tone = get_tone_instruction(rank)
    
    sys_msg = f"""You are an objective but discerning HR analyst evaluating candidates. Draft a crisp, 2-sentence summary of the candidate.
                Your response MUST strictly follow this exact flow and be exactly between 30 and 40 words:
                Sentence 1: State the last title, years of experience, key skills, and industry.
                Sentence 2: State the experience status, and end the sentence strictly with the notice period and immediate hiring eligibility.

                CRITICAL RULES:
                - {tone} Incorporate this tone into your word choices naturally without sounding robotic.
                - Vary your sentence starters: Use "The candidate" for one sentence, and the job title (e.g., "The AI Engineer") for the other. 
                - Do NOT start any sentence with a pronoun (He, She, They). You may use singular pronouns (his, her) in the middle of sentences.
                - Avoid using "they" or "their" since the candidate is singular. Ensure proper singular verb agreement.
                - NO standalone conclusions (e.g., do not add "This makes them a good fit" at the very end). 
                - End the text immediately after stating the notice period.
                - Write in neutral third person. Do not use "we", "our", or "us".
                """

    prompt = f"""<|start_header_id|>system<|end_header_id|>
                {sys_msg} Output ONLY the final reasoning text.<|eot_id|><|start_header_id|>user<|end_header_id|>
                Title: {title}
                Experience: {exp} years
                Skills: {skills}
                Industry: {industry}
                Experience Note: {exp_gap}
                Notice Period: {notice} days ({is_immediate})
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """

    return prompt

# Output Cleanser
def enforce_sentence_limit(text):
    text = re.sub(r'(?i)^(here is( the)?.*?:|based on.*?:|title:\s*|experience:\s*)', '', text).strip()
    text = re.sub(r'(?i)^our (team|company) (is seeking|seeks to hire|has hired|seeks) (for )?(a|an|the )?', '', text).strip()
    
    replacements = {
        r'\b(?:i|they)\b': 'the candidate',
        r'\b(?:my|their)\b': "the candidate's",
        r'\b(?:we)\b': 'the organization',
        r'\b(?:our)\b': "the organization's",
        r'\b(?:you)\b': 'the candidate',
        r'\b(?:your)\b': "the candidate's",
        r'\b(?:John Doe|John|Johnson|Ms\. Smith|Mr\. Smith|Smith)\b': 'the candidate'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    verb_fixes = {
        r'(?i)\b(the candidate|the candidate\'s)\s+are\b': r'\1 is',
        r'(?i)\b(the candidate|the candidate\'s)\s+have\b': r'\1 has',
        r'(?i)\b(the candidate|the candidate\'s)\s+were\b': r'\1 was',
        r'(?i)\b(the candidate|the candidate\'s)\s+do\b': r'\1 does',
        r'(?i)\b(the candidate|the candidate\'s)\s+lack\b': r'\1 lacks',
        r'(?i)\b(the candidate|the candidate\'s)\s+hold\b': r'\1 holds',
        r'(?i)\b(the candidate|the candidate\'s)\s+possess\b': r'\1 possesses',
        r'(?i)\b(the candidate|the candidate\'s)\s+show\b': r'\1 shows',
        r'(?i)\b(the candidate|the candidate\'s)\s+demonstrate\b': r'\1 demonstrates',
        r'(?i)\b(the candidate|the candidate\'s)\s+need\b': r'\1 needs'
    }
    for pattern, replacement in verb_fixes.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'[\n\r]+', ' ', text)
    text = text.replace('"', '').replace("'", "").strip()
    
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    
    # Target only subjective conclusions
    fluff_pattern = re.compile(r'(?i)[,\s]*(making the candidate|thus|therefore|which aligns|suggesting|highlighting|demonstrating|showcasing|meaning|resulting in|indicating a good fit).*$')
    
    processed_sentences = []
    for s in valid_sentences[:2]:
        s = fluff_pattern.sub('', s)
        
        # Enforce no pronouns at the very start of the sentence
        s = re.sub(r'^(He|She|It)\b\s*', 'The candidate ', s, flags=re.IGNORECASE)
        
        if s:
            s = s[0].upper() + s[1:]
        processed_sentences.append(s)

    final_text = " ".join(processed_sentences).strip()
    
    if final_text and not final_text.endswith(('.', '!', '?')):
        final_text = re.sub(r'[,;:\-]\s*$', '', final_text) + '.'
        
    return final_text

# Top level export
def generate_reasoning(candidate, rank):
    title, exp, skills, notice, is_immediate, exp_gap, industry = extract_candidate_facts(candidate)
    prompt = get_hr_prompt(rank, title, exp, skills, notice, is_immediate, exp_gap, industry)
    
    llm = get_llm()
    
    response = llm(
        prompt,
        max_tokens=90,
        temperature=0.65,
        top_p=0.92,
        repeat_penalty=1.15,  
        stop=["<|eot_id|>"]
    )
    
    raw_text = response['choices'][0]['text'].strip()
    return enforce_sentence_limit(raw_text)