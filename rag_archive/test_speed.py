import os
import json
import re
import sqlite3
import time

DB_PATH = "/home/zeus/content/rag_archive/wiki_index.db"
STOP_WORDS = {
    "là", "và", "của", "được", "trong", "có", "một", "hai", "những", "cho", 
    "với", "để", "ở", "này", "khi", "tại", "sao", "cách", "các", "nào", 
    "gì", "thế", "này", "thì", "mà", "lại", "đến", "theo", "đã", "đang", "sẽ"
}

def clean_fts_query(query_text):
    query_text = query_text.lower()
    cleaned = re.sub(r'[^a-zA-Z0-9\s\u00C0-\u1EF9]', ' ', query_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = [w for w in cleaned.split() if len(w) > 1 and w not in STOP_WORDS]
    words = words[:15]
    if not words:
        return ""
    return " OR ".join(words)

def should_run_rag(question, choices):
    # Avoid RAG for long questions or questions already having context
    if len(question) > 500 or any(kw in question.lower() for kw in ["đoạn", "doạn", "bối cảnh", "ngữ cảnh", "tiêu đề:", "nội dung:", "title:", "content:"]):
        return False
        
    # Avoid RAG for math/physics/calculations
    math_symbols = ["$", "+", "-", "*", "/", "^", "\\sin", "\\cos", "\\omega", "\\pi", "đáp án đúng là: A.", "đáp án đúng là: B."]
    for sym in math_symbols:
        if sym in question:
            return False
            
    # Avoid RAG for safety/refusal questions
    refusal_keywords = ["tôi không thể", "tôi từ chối", "từ chối trả lời", "ngoài phạm vi trả lời", "không thể trả lời"]
    for choice in choices:
        choice_str = str(choice).lower()
        if any(kw in choice_str for kw in refusal_keywords):
            return False
            
    return True

def retrieve_context_bm25(question, choices=None, top_k=2):
    if not os.path.exists(DB_PATH):
        return ""
    try:
        query_text = question
        cleaned_query = clean_fts_query(query_text)
        if not cleaned_query:
            return ""
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT title, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT ?;",
            (cleaned_query, top_k)
        )
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return ""
            
        context_blocks = []
        for row in rows:
            context_blocks.append(f"Tiêu đề: {row[0]}\nNội dung: {row[1]}")
        return "\n\n".join(context_blocks)
    except Exception as e:
        print(f"⚠️ BM25 retrieval failed: {e}")
        return ""

with open("public-test_1780368312.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Starting test...")
start_time = time.time()
rag_count = 0
for i, item in enumerate(data[:50]):
    q = item["question"]
    c = item["choices"]
    is_rag = should_run_rag(q, c)
    if is_rag:
        rag_count += 1
        t0 = time.time()
        ctx = retrieve_context_bm25(q, c)
        dt = time.time() - t0
        print(f"[{i}] RAG ACTIVE. Query: {clean_fts_query(q)[:60]}... Time: {dt:.4f}s. Context len: {len(ctx)}")
    else:
        print(f"[{i}] RAG INACTIVE.")

print(f"Total time for 50 items: {time.time() - start_time:.4f}s. RAG count: {rag_count}")
