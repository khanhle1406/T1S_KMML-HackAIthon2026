import re
import sqlite3
import time
import json

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
    if not words:
        return ""
    return " OR ".join(words)

def retrieve_context_bm25(question, choices=None, top_k=2):
    t0 = time.time()
    query_text = question
    if choices:
        query_text += " " + " ".join([str(c) for c in choices])
        
    cleaned_query = clean_fts_query(query_text)
    print(f"Cleaned query length (chars): {len(cleaned_query)}")
    print(f"Number of words: {len(cleaned_query.split(' OR '))}")
    
    if not cleaned_query:
        return ""
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        t1 = time.time()
        cursor.execute(
            "SELECT title, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT ?;",
            (cleaned_query, top_k)
        )
        rows = cursor.fetchall()
        t2 = time.time()
        print(f"Query executed in {t2 - t1:.4f} seconds, found {len(rows)} rows")
        return rows
    except Exception as e:
        print(f"Error during query execution: {e}")
        return []
    finally:
        conn.close()

# Let's load the first question from the JSON file
with open("public-test_1780368312.json", "r", encoding="utf-8") as f:
    data = json.load(f)

item = data[0]
print("Running retrieve_context_bm25 on first item...")
retrieve_context_bm25(item["question"], item["choices"])
