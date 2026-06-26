import re
import sqlite3
import time

DB_PATH = "/home/zeus/content/rag_archive/wiki_index.db"
STOP_WORDS = {
    "là", "và", "của", "được", "trong", "có", "một", "hai", "những", "cho", 
    "với", "để", "ở", "này", "khi", "tại", "sao", "cách", "các", "nào", 
    "gì", "thế", "này", "thì", "mà", "lại", "đến", "theo", "đã", "đang", "sẽ"
}

def clean_fts_query(query_text, op="OR", limit=15):
    query_text = query_text.lower()
    cleaned = re.sub(r'[^a-zA-Z0-9\s\u00C0-\u1EF9]', ' ', query_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = [w for w in cleaned.split() if len(w) > 1 and w not in STOP_WORDS]
    words = words[:limit]
    if not words:
        return ""
    if op == "OR":
        return " OR ".join(words)
    else:
        # FTS5 implicit AND
        return " ".join(words)

def test_query(query_text, op, limit):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    q = clean_fts_query(query_text, op, limit)
    t0 = time.time()
    try:
        cursor.execute(
            "SELECT title, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT 2;",
            (q,)
        )
        rows = cursor.fetchall()
        dt = time.time() - t0
        print(f"[{op} limit={limit}] Time: {dt:.4f}s, Results: {len(rows)}, Query: {q[:80]}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

# Sample query: "Một trong các biểu hiện của biến đổi khí hậu là gì?"
q = "Một trong các biểu hiện của biến đổi khí hậu là gì?"
print("Test query:", q)
test_query(q, "OR", 15)
test_query(q, "OR", 5)
test_query(q, "AND", 5)
test_query(q, "AND", 3)

# Another one:
q2 = "Nếu bảng cầu của một sản phẩm cho thấy tại mức giá 5 đô la, lượng cầu là 150 đơn vị, và tại mức giá 3 đô la, lượng cầu là 250 đơn vị, thì độ co giãn của cầu theo giá giữa hai điểm này là bao nhiêu?"
print("\nTest query 2:", q2[:100])
test_query(q2, "OR", 15)
test_query(q2, "OR", 5)
test_query(q2, "AND", 5)
test_query(q2, "AND", 3)
