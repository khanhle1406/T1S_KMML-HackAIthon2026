import os
import json
import re
import sqlite3

DB_PATH = "/home/zeus/content/rag_archive/wiki_index.db"
STOP_WORDS = {
    "là", "và", "của", "được", "trong", "có", "một", "hai", "những", "cho", 
    "với", "để", "ở", "này", "khi", "tại", "sao", "cách", "các", "nào", 
    "gì", "thế", "này", "thì", "mà", "lại", "đến", "theo", "đã", "đang", "sẽ"
}

def select_best_keywords(query_text, max_keywords=8):
    raw_words = query_text.split()
    if not raw_words:
        return []
    
    processed_words = []
    for i, w in enumerate(raw_words):
        w_clean = re.sub(r'[^a-zA-Z0-9\u00C0-\u1EF9]', '', w)
        if len(w_clean) <= 1:
            continue
        w_lower = w_clean.lower()
        if w_lower in STOP_WORDS:
            continue
            
        is_cap = w[0].isupper() if w else False
        if i == 0:
            is_cap = False 
            
        score = len(w_clean)
        if is_cap:
            score += 10
            
        processed_words.append((w_lower, score, i))
        
    processed_words.sort(key=lambda x: (-x[1], x[2]))
    selected = processed_words[:max_keywords]
    selected.sort(key=lambda x: x[2])
    return [x[0] for x in selected]

def clean_fts_query(query_text, op="AND", limit=8):
    words = select_best_keywords(query_text, max_keywords=limit)
    if not words:
        return ""
    if op == "OR":
        return " OR ".join(words)
    return " ".join(words)

def should_run_rag(question, choices):
    # 1. Avoid RAG for long questions or questions already having context
    if len(question) > 400 or any(kw in question.lower() for kw in ["đoạn", "doạn", "bối cảnh", "ngữ cảnh", "tiêu đề:", "nội dung:", "title:", "content:"]):
        return False
        
    # 2. Avoid RAG for math/physics/chemistry/finance calculations
    math_symbols = ["$", "+", "-", "*", "/", "^", "=", "\\sin", "\\cos", "\\omega", "\\pi"]
    question_lower = question.lower()
    for sym in math_symbols:
        if sym in question_lower:
            return False
            
    math_patterns = [
        r'\b\d+\s*(?:ml|m|g|kg|lít|lit|cm|mm|dm|km|m²|m3|cm³|m/s|m/s2|m/s²|%|usd|đô la|dollar|vnd|đồng|tỷ|triệu)\b', # numbers with units
        r'\b(?:naoh|hcl|h2so4|ch3cooh|co2|o2|h2o|c6h12o6)\b',
        r'\b(?:gdp|gnp|cpi)\b',
        r'\b(?:hàm số|phương trình|đồ thị|tích phân|đạo hàm|tiệm cận|giới hạn|tích vô hướng|vectơ|đường tròn|đường thẳng|tam giác|hình chóp|lực kế|nhiệt lượng|nhiệt độ|nhiệt dung|nhiệt trở|điện trở|điện áp|dòng điện|vận tốc|gia tốc|chu kỳ|tần số|tần số góc|biên độ|pha dao động|thế năng|động năng|cơ năng)\b',
        r'\b(?:lãi suất|tiết kiệm|kỳ hạn|doanh thu|chi phí|sản lượng|lợi nhuận|lạm phát|cung tiền|cầu theo giá|hàm sản xuất|co giãn của cầu|lượng cầu|độ co giãn)\b',
        r'\b(?:xác suất|ngẫu nhiên|kỳ vọng|phương sai|độ lệch chuẩn|kiểm định|giả thuyết|mức ý nghĩa|độ tin cậy)\b'
    ]
    for pattern in math_patterns:
        if re.search(pattern, question_lower):
            return False
            
    # 3. Avoid RAG for safety/refusal questions
    refusal_keywords = ["tôi không thể", "tôi từ chối", "từ chối trả lời", "ngoài phạm vi trả lời", "không thể trả lời"]
    for choice in choices:
        choice_str = str(choice).lower()
        if any(kw in choice_str for kw in refusal_keywords):
            return False
            
    return True

def retrieve_context_bm25(question, choices=None, top_k=2):
    if not os.path.exists(DB_PATH):
        return "NO_DB"
    
    # Thử với AND trước (chính xác cao, nhanh)
    q_and = clean_fts_query(question, op="AND", limit=8)
    if not q_and:
        return ""
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT title, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT ?;",
            (q_and, top_k)
        )
        rows = cursor.fetchall()
        
        # Nếu không có kết quả, thử với OR (rộng hơn) nhưng giới hạn số từ nhỏ để chạy nhanh
        if not rows:
            q_or = clean_fts_query(question, op="OR", limit=8)
            cursor.execute(
                "SELECT title, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT ?;",
                (q_or, top_k)
            )
            rows = cursor.fetchall()
            
        if not rows:
            return ""
            
        context_blocks = []
        for row in rows:
            context_blocks.append(f"Tiêu đề: {row[0]}\nNội dung: {row[1]}")
        return "\n\n".join(context_blocks)
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        conn.close()

with open("public-test_1780368312.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output = []
for i, item in enumerate(data[:50]):
    q = item["question"]
    c = item["choices"]
    is_rag = should_run_rag(q, c)
    ctx = ""
    if is_rag:
        ctx = retrieve_context_bm25(q, c)
    
    output.append({
        "qid": item["qid"],
        "is_rag": is_rag,
        "question_prefix": q[:80].replace("\n", " "),
        "context_preview": ctx[:150].replace("\n", " ") if ctx else ""
    })

with open("inspect_output.json", "w", encoding="utf-8") as f_out:
    json.dump(output, f_out, ensure_ascii=False, indent=2)
print("Done writing inspect_output.json")
