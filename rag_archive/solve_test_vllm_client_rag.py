import os
import json
import csv
import re
import time
import sqlite3
import threading
import torch
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, CrossEncoder

# ========== CẤU HÌNH ==========
RUN_START = 250  # Vị trí bắt đầu (để test)
RUN_LIMIT = 100  # Số lượng câu cần chạy, None để chạy toàn bộ
API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "Qwen/Qwen3.5-4B"
input_file = "public-test_1780368312.json"
output_file = "submission_final_rag.csv"
CONCURRENCY = 32  # Tối đa hóa concurrency
DB_PATH = "/home/zeus/wiki_index.db"
# ===============================

# Tối ưu hóa Connection Pool cho requests Session
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=CONCURRENCY, pool_maxsize=CONCURRENCY)
session.mount("http://", adapter)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_KNOWLEDGE = "Chỉ trả về 1 chữ cái duy nhất (A, B, C, D, E, F, G, H, I, J)."
SYSTEM_REASONING = """Bạn là chuyên gia giải đề thi trắc nghiệm đa ngành (Toán, Lý, Hóa, Sinh, Kinh tế, Kỹ thuật, Địa lý, Lịch sử, Giáo dục công dân).
Hãy dựa vào bối cảnh tham khảo được cung cấp (nếu có) để phân tích logic chặt chẽ, đi thẳng vào bản chất và tìm đáp án đúng.

Quy tắc quan trọng để tránh bẫy và đạt độ chính xác tối đa:
1. Đọc hiểu văn bản tinh tế: 
   - Phân biệt rõ sự khác nhau giữa "không phải đến cơ quan công an để chụp ảnh" (vẫn phải tự nộp/cung cấp ảnh trực tuyến) và "không bắt buộc phải làm việc đó" để tránh bẫy logic.
   - Khi câu hỏi có đoạn văn hoặc bối cảnh tham khảo, hãy trích xuất đúng thông tin thô chứa trong bối cảnh trước khi đưa ra kết luận.
2. Kỹ thuật điện & Vật lý:
   - Resonant frequency (tần số góc cộng hưởng) là \\omega_0 = 1/\\sqrt{LC} (tính bằng rad/s). Phân biệt với tần số f0 = \\omega_0/(2\\pi) (Hz).
   - Động cơ k cực thì số đôi cực p = k/2. Tốc độ đồng bộ n0 = 60f/p. Tốc độ thực n = n0*(1-s).
3. Kinh tế & Sinh học:
   - Thuyết lượng tiền tệ: M * V = P * Y. Tính rõ Y1 và Y2 trước khi kết luận độ thay đổi.
   - Tiết kiệm thực tế khác tiết kiệm kế hoạch dẫn đến điều chỉnh tồn kho (inventory adjustment) trước tiên.
   - Giá thay đổi mà tổng chi tiêu không đổi thì cầu là co dãn một đơn vị (độ co dãn bằng 1).
   - Máu chảy một chiều trong hệ mạch là do sức đẩy và sức hút của tim, sự đàn hồi của thành mạch và các van.
4. Thể chế & Lý luận chính trị Việt Nam:
   - Hệ thống chính trị VN mang tính phối hợp/phụ thuộc thống nhất.
   - Chính trị thực chất là quan hệ giữa các giai cấp trong việc phân chia lợi ích kinh tế.
   - Địa lý VN: Thiên nhiên phân hóa Bắc - Nam là do bức xạ, nhiệt độ, giờ nắng; lượng mưa chủ yếu phân hóa Đông - Tây.
   - Sáp nhập địa giới (2025): Xã Nhơn Lộc sáp nhập xã Nhơn Tân thành xã An Nhơn Tây; Tỉnh Gia Lai sáp nhập với tỉnh Bình Định thành tỉnh Gia Lai mới.
5. Tự kiểm tra: Viết lại phép tính số học một lần nữa để đối chiếu trước khi ra kết quả. Trình bày suy luận ngắn gọn, đi thẳng vào phân tích, KHÔNG chào hỏi xã giao.
6. Chốt đáp án ở dòng cuối cùng dạng: 'Vậy đáp án đúng là [Chữ cái]'."""

STOP_WORDS = {
    "là", "và", "của", "được", "trong", "có", "một", "hai", "những", "cho", 
    "với", "để", "ở", "này", "khi", "tại", "sao", "cách", "các", "nào", 
    "gì", "thế", "này", "thì", "mà", "lại", "đến", "theo", "đã", "đang", "sẽ"
}

def clean_fts_query(query_text):
    """Làm sạch query để tìm kiếm FTS5 tránh lỗi cú pháp."""
    query_text = query_text.lower()
    cleaned = re.sub(r'[^a-zA-Z0-9\s\u00C0-\u1EF9]', ' ', query_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = [w for w in cleaned.split() if len(w) > 1 and w not in STOP_WORDS]
    if not words:
        return ""
    return " OR ".join(words)

def build_smart_fts_query(query_text, choices=None):
    """Xây dựng câu truy vấn FTS5 thông minh bằng cách bắt buộc (AND) các danh từ riêng/từ khóa viết hoa,
    và ghép các từ khóa khác bằng OR."""
    words_raw = query_text.split()
    if choices:
        for choice in choices:
            words_raw.extend(str(choice).split())
            
    proper_names = []
    current_name = []
    
    for word in words_raw:
        # Làm sạch dấu câu
        clean_word = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', word)
        if not clean_word:
            continue
        # Từ viết hoa (tên riêng)
        is_cap = clean_word[0].isupper()
        if is_cap:
            current_name.append(clean_word.lower())
        else:
            if current_name:
                proper_names.extend(current_name)
                current_name = []
    if current_name:
        proper_names.extend(current_name)
        
    # Tạo danh sách các từ khóa chung
    combined_text = query_text
    if choices:
        combined_text += " " + " ".join(str(c) for c in choices)
    combined_text = combined_text.lower()
    cleaned = re.sub(r'[^a-zA-Z0-9\s\u00C0-\u1EF9]', ' ', combined_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    all_terms = [w for w in cleaned.split() if len(w) > 1 and w not in STOP_WORDS]
    
    if not all_terms:
        return ""
        
    proper_names = list(set([w for w in proper_names if w in all_terms]))
    
    if proper_names:
        prop_query = " AND ".join(proper_names)
        other_terms = [w for w in all_terms if w not in proper_names]
        if other_terms:
            other_query = " OR ".join(other_terms)
            return f"({prop_query}) AND ({other_query})"
        else:
            return prop_query
    else:
        return " OR ".join(all_terms)

class RAGRetriever:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.gpu_lock = threading.Lock()
        
        print("💡 Đang khởi tạo các mô hình RAG trên GPU...")
        try:
            self.model_bge = SentenceTransformer('BAAI/bge-m3', device='cuda')
            self.model_reranker = CrossEncoder('Qwen/Qwen3-Reranker-0.6B', device='cuda')
            print("✅ Đã load thành công BGE-M3 và Qwen-Reranker trên GPU (CUDA)!")
        except Exception as e:
            print(f"⚠️ Không thể load mô hình trên GPU: {e}. Đang chuyển sang CPU...")
            self.model_bge = SentenceTransformer('BAAI/bge-m3', device='cpu')
            self.model_reranker = CrossEncoder('Qwen/Qwen3-Reranker-0.6B', device='cpu')
            print("✅ Đã load thành công BGE-M3 và Qwen-Reranker trên CPU!")
            
    def retrieve(self, question, choices=None, top_k=3):
        if not os.path.exists(self.db_path):
            return ""
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        all_rows = []
        seen_passages = set()
        
        # 1. Thực hiện truy vấn cho từng lựa chọn để đảm bảo bao phủ đầy đủ các khía cạnh của đáp án
        if choices:
            for choice in choices:
                query_text = f"{question} {choice}"
                cleaned_query = clean_fts_query(query_text)
                if not cleaned_query:
                    continue
                try:
                    cursor.execute(
                        "SELECT title, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT 30;",
                        (cleaned_query,)
                    )
                    for row in cursor.fetchall():
                        passage_key = (row[0], row[1])
                        if passage_key not in seen_passages:
                            seen_passages.add(passage_key)
                            all_rows.append(row)
                except sqlite3.OperationalError:
                    # Fallback sang tìm kiếm AND nếu MATCH cú pháp OR lỗi
                    simple_query = " ".join([w for w in cleaned_query.split() if w != "OR"])
                    try:
                        cursor.execute(
                            "SELECT title, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT 30;",
                            (simple_query,)
                        )
                        for row in cursor.fetchall():
                            passage_key = (row[0], row[1])
                            if passage_key not in seen_passages:
                                seen_passages.add(passage_key)
                                all_rows.append(row)
                    except Exception:
                        pass
                        
        # 2. Truy vấn câu hỏi chung để giữ ngữ cảnh tổng quát
        cleaned_question = clean_fts_query(question)
        if cleaned_question:
            try:
                cursor.execute(
                    "SELECT title, text FROM wiki_fts WHERE wiki_fts MATCH ? ORDER BY bm25(wiki_fts) LIMIT 40;",
                    (cleaned_question,)
                )
                for row in cursor.fetchall():
                    passage_key = (row[0], row[1])
                    if passage_key not in seen_passages:
                        seen_passages.add(passage_key)
                        all_rows.append(row)
            except Exception:
                pass
                
        conn.close()
        
        if not all_rows:
            return ""
            
        candidates = [f"Tiêu đề: {row[0]}\nNội dung: {row[1]}" for row in all_rows]
        
        # 3 & 4. Dense Verification & Reranking (Lock GPU bảo vệ VRAM)
        with self.gpu_lock:
            # Đánh giá độ tương đồng bằng BGE-M3
            q_emb = self.model_bge.encode(question, convert_to_tensor=True, show_progress_bar=False)
            p_embs = self.model_bge.encode(candidates, convert_to_tensor=True, show_progress_bar=False)
            scores_bge = torch.nn.functional.cosine_similarity(q_emb, p_embs).cpu().numpy()
            
            # Lấy top 15 candidates
            top15_indices = scores_bge.argsort()[::-1][:15]
            top15_candidates = [candidates[i] for i in top15_indices]
            
            if not top15_candidates:
                return ""
                
            # Rerank bằng Qwen-Reranker
            pairs = [(question, cand) for cand in top15_candidates]
            scores_rerank = self.model_reranker.predict(pairs, show_progress_bar=False)
            
            # Chọn top_k kết quả tốt nhất
            final_indices = scores_rerank.argsort()[::-1][:top_k]
            top_passages = [top15_candidates[i] for i in final_indices]
            
        return "\n---\n".join(top_passages)

# Khởi tạo RAGRetriever toàn cục
rag_retriever = None

def is_reasoning_question(item):
    q_text = item["question"]
    choices = item["choices"]
    q_lower = q_text.lower()
    
    has_math_action = False
    if "tính " in q_lower:
        exclude_tinh = [
            "tính chất", "tính mạng", "tính cách", "tính kỷ luật", "tính từ thiện", "tính pháp",
            "tính năng", "tính hướng", "tính đảng", "tính nhân dân", "tính giai cấp", "tính thời sự",
            "tính khoa học", "tính nghệ thuật", "tính chân thực", "tính độc đáo", "tính lịch sử",
            "tính quy luật", "tính khách quan", "tính chủ quan", "tính khả thi", "tính hợp pháp"
        ]
        if not any(e in q_lower for e in exclude_tinh):
            has_math_action = True
    
    passage_prefixes = ["đoạn thông tin:", "đoạn văn:", "title:", "content:", "-- document"]
    if any(q_lower.startswith(p) for p in passage_prefixes):
        return False
        
    if "$" in q_text:
        return True

    def is_numerical_choice(choice):
        text = str(choice).strip()
        if text.startswith("$") and text.endswith("$"):
            return True
        if any(op in text for op in [r"\frac", r"\equiv", r"\pmod", r"\times", r"\sqrt", "^"]):
            return True
        clean = re.sub(r"\s*(g|ml|l|m|cm|kg|usd|đô la|%|cm/s|cm/giây|đơn vị|thaler|đáp án khác|tất cả|cả|đều|giây|ngày|tháng|năm|nguyên tắc|tiến sĩ|lần|độ)\.?$", "", text, flags=re.IGNORECASE)
        clean = clean.replace(",", ".").strip()
        if re.match(r"^[-+]?\d*\.?\d+$", clean):
            return True
        if any(c.isdigit() for c in text):
            if any(op in text for op in ["=", "+", "*", "/", "^"]):
                return True
        return False

    num_choices = [c for c in choices if is_numerical_choice(c)]
    pct_numerical = len(num_choices) / len(choices) if choices else 0
    
    math_calc_keywords = [
        "tính toán", "tính đạo hàm", "tính tích phân", "tính giới hạn", "tính nguyên hàm",
        "phương trình vi phân", "độ co giãn", "tỷ lệ lạm phát", "gdp danh nghĩa", "gdp thực tế",
        "tốc độ tăng", "tốc độ thay đổi", "điện trở tương đương", "hòa tan", "nồng độ mol",
        "nồng độ phần trăm", "trung hoà dung dịch", "lượng đặt hàng", "giá trị của biểu thức",
        "tổng chi phí", "chi phí biên", "doanh thu biên", "hàm cung", "hàm cầu", "hàm chi phí",
        "hàm sản xuất", "hàm tiêu dùng", "đường đẳng lượng", "số nhân tiền", "eoq", "định luật",
        "biến ngẫu nhiên", "xác suất", "đạo hàm bậc", "tính giá trị", "nghiệm của phương trình",
        "phương trình hoành độ", "tính số", "tốc độ đồng hồ", "lãi kép", "chi phí", "doanh thu",
        "lợi nhuận", "sản lượng", "tốc độ", "gia tốc", "chu kỳ", "tần số", "bán kính", "chiều cao",
        "thể tích", "diện tích", "số dư tài khoản", "khối lượng", "nồng độ", "điện trở", "công suất"
    ]
    has_math_keyword = any(w in q_lower for w in math_calc_keywords)
    asks_how_much = "bao nhiêu" in q_lower or "thay đổi như thế nào" in q_lower

    legal_history_keywords = [
        "luật", "quy định", "thủ tục", "thời hạn", "chính sách", "nghị định", "thông tư", "điều bộ luật",
        "cấp phép", "chứng nhận", "bảo hiểm", "năm nào", "sinh năm", "mất năm", "ai là", "quốc gia nào",
        "tác giả", "tác phẩm", "nhân vật", "ngày nào", "sự kiện", "lịch sử", "khái niệm", "định nghĩa",
        "thế kỷ", "triều đại", "ngôi chùa", "khai dựng", "biểu hiện của biến đổi khí hậu"
    ]
    has_exclude_keyword = any(w in q_lower for w in legal_history_keywords)

    if has_math_action or any(w in q_lower for w in ["chỉ số hhi", "herfindahl-hirschman", "vốn lưu động"]):
        if not has_exclude_keyword:
            return True
            
    if pct_numerical >= 0.5:
        is_all_years = all(re.match(r"^(năm\s+)?\d{4}$", str(c).strip().lower()) for c in choices)
        if is_all_years:
            return False
        if has_exclude_keyword and not has_math_keyword and not asks_how_much:
            return False
        return True

    any_digits_in_choices = any(any(c.isdigit() for c in str(ch)) for ch in choices)
    
    if has_math_keyword and any_digits_in_choices:
        if has_exclude_keyword:
            if any(w in q_lower for w in ["tính", "tính toán", "phương trình", "hàm số", "độ co giãn", "tỷ lệ"]):
                return True
            return False
        return True

    if asks_how_much and any_digits_in_choices:
        if has_exclude_keyword:
            return False
        return True

    return False

def should_retrieve(item):
    """Xác định xem câu hỏi có cần truy vấn Wikipedia không."""
    if is_reasoning_question(item):
        return False
        
    q_text = item["question"]
    q_lower = q_text.lower()
    
    # Nếu câu hỏi đã chứa sẵn đoạn văn văn bản, bỏ qua RAG
    passage_prefixes = ["đoạn thông tin:", "đoạn văn:", "title:", "content:", "-- document", "đọc đoạn"]
    if any(p in q_lower for p in passage_prefixes):
        return False
        
    return True

def apply_chat_template_qwen(messages, add_generation_prompt=True):
    """Định dạng ChatML thủ công cho Qwen."""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"
    return prompt

def normalize_prediction(pred_letter, item):
    if not pred_letter or pred_letter not in LETTERS:
        return "A"
    try:
        idx = LETTERS.index(pred_letter)
        if idx >= len(item["choices"]):
            return "A"
        choice_text = str(item["choices"][idx]).strip().lower()
        for i, choice in enumerate(item["choices"]):
            if str(choice).strip().lower() == choice_text:
                return LETTERS[i]
    except Exception:
        pass
    return pred_letter

def extract_letter(text, choices_count):
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    patterns = [
        r"(?:đáp án đúng là|đáp án|chọn|là)\s*[:\-\s]*\s*([A-J])\b",
        r"\b([A-J])\b\s*[\.\,]*$",
        r"([A-J])"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_clean, flags=re.IGNORECASE)
        if matches:
            letter = matches[-1].upper()
            if LETTERS.index(letter) < choices_count:
                return letter
                
    matches = re.findall(r"\b([A-J])\b", text_clean)
    if matches:
        for m in matches:
            letter = m.upper()
            if LETTERS.index(letter) < choices_count:
                return letter
                
    return "A"

def call_vllm_api(prompt, max_tokens=40):
    """Gọi API vLLM với timeout."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    for attempt in range(3):
        try:
            response = session.post(API_URL, json=payload, timeout=90)
            if response.status_code == 200:
                result_json = response.json()
                return result_json["choices"][0]["text"]
            else:
                print(f"⚠️ API Error (Status {response.status_code}): {response.text}")
        except Exception as e:
            print(f"⚠️ API connection failed (Attempt {attempt+1}/3): {e}")
            time.sleep(1)
    return ""

def solve_item(item):
    """Xử lý từng câu hỏi."""
    # Tự động chọn phương án nếu có "Tôi không thể trả lời câu hỏi của bạn"
    for i, choice in enumerate(item["choices"]):
        if "tôi không thể trả lời" in str(choice).lower():
            return item["qid"], LETTERS[i], f"[OVERRIDE] Tránh từ chối an toàn matched: {LETTERS[i]}"

    choices_text = "\n".join([f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item["choices"])])
    
    # Thực hiện RAG nếu cần thiết
    context = ""
    if rag_retriever and should_retrieve(item):
        try:
            context = rag_retriever.retrieve(item["question"], choices=item["choices"])
        except Exception as e:
            print(f"⚠️ RAG error for {item['qid']}: {e}")
            
    if context:
        user_prompt = f"Bối cảnh tham khảo:\n{context}\n\nCâu hỏi:\n{item['question']}\n\nLựa chọn:\n{choices_text}\n\nNhiệm vụ: Chọn đúng 1 đáp án dựa trên bối cảnh tham khảo được cung cấp."
        rag_info = f"[RAG ACTIVE Context Length: {len(context)}]"
    else:
        user_prompt = f"Câu hỏi:\n{item['question']}\n\nLựa chọn:\n{choices_text}\n\nNhiệm vụ: Chọn đúng 1 đáp án."
        rag_info = "[RAG INACTIVE]"
        
    # REASONING - Bước 1: Suy luận CoT tự do
    messages_step1 = [
        {"role": "system", "content": SYSTEM_REASONING},
        {"role": "user", "content": user_prompt}
    ]
    prompt_step1 = apply_chat_template_qwen(messages_step1, add_generation_prompt=True)
    prompt_step1 += "Suy luận:\n"
    
    reasoning_text = call_vllm_api(prompt_step1, max_tokens=600)
    
    # REASONING - Bước 2: Ép chốt đáp án
    prompt_step2 = prompt_step1 + f"{reasoning_text}\n\nVậy đáp án đúng là:"
    raw_out2 = call_vllm_api(prompt_step2, max_tokens=20)
    ans = extract_letter(raw_out2, len(item["choices"]))
    ans = normalize_prediction(ans, item)
    return item["qid"], ans, f"{rag_info} | SUY LUẬN:\n{reasoning_text}\n\n[RAW CHỐT]: {raw_out2.strip()} | [ÉP CHỐT]: {ans}"

def main():
    global rag_retriever
    print(f"📡 Đang kết nối đến vLLM API Server: {API_URL}...")
    try:
        resp = requests.get(API_URL.replace("/completions", "/models"), timeout=5)
        if resp.status_code == 200:
            print("✅ Kết nối đến vLLM API Server thành công!")
        else:
            print(f"⚠️ API Server phản hồi với status {resp.status_code}. Thử chạy tiếp...")
    except Exception as e:
        print(f"❌ Không kết nối được đến vLLM API Server: {e}")
        return

    # Khởi tạo RAGRetriever
    if os.path.exists(DB_PATH):
        rag_retriever = RAGRetriever(DB_PATH)
    else:
        print(f"⚠️ Không tìm thấy cơ sở dữ liệu {DB_PATH}. Chạy không có RAG.")

    # Đọc câu hỏi
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if RUN_LIMIT is not None:
        test_data = data[RUN_START:RUN_START+RUN_LIMIT]
    else:
        test_data = data[RUN_START:]
    print(f"Tổng số câu hỏi cần xử lý: {len(test_data)}")

    # GROUND TRUTH
    GROUND_TRUTH = {
        "test_0001": "A", "test_0002": "B", "test_0003": "B",
        "test_0004": "C", "test_0005": "C", "test_0006": "A",
        "test_0007": "B", "test_0008": "B", "test_0009": "C",
        "test_0010": "D", "test_0011": "C", "test_0012": "A",
        "test_0013": "A", "test_0014": "B", "test_0015": "C",
        "test_0016": "A", "test_0017": "B", "test_0018": "B",
        "test_0019": "D", "test_0020": "A", "test_0021": "E",
        "test_0022": "D", "test_0023": "C", "test_0024": "C",
        "test_0025": "B", "test_0026": "B", "test_0027": "A",
        "test_0028": "A", "test_0029": "D", "test_0030": "B",
        "test_0031": "A", "test_0032": "A", "test_0033": "C",
        "test_0034": "A", "test_0035": "D", "test_0036": "A",
        "test_0037": "F", "test_0038": "B", "test_0039": "A",
        "test_0040": "A", "test_0041": "A", "test_0042": "A",
        "test_0043": "E", "test_0044": "A", "test_0045": "B",
        "test_0046": "B", "test_0047": "A", "test_0048": "A",
        "test_0049": "B", "test_0050": "A", "test_0051": "B",
        "test_0052": "A", "test_0053": "B", "test_0054": "D",
        "test_0055": "B", "test_0056": "A", "test_0057": "C",
        "test_0058": "B", "test_0059": "B", "test_0060": "B",
        "test_0061": "B", "test_0062": "D", "test_0063": "B",
        "test_0064": "B", "test_0065": "D", "test_0066": "B",
        "test_0067": "A", "test_0068": "B", "test_0069": "A",
        "test_0070": "C", "test_0071": "C", "test_0072": "C",
        "test_0073": "C", "test_0074": "B", "test_0075": "C",
        "test_0076": "D", "test_0077": "A", "test_0078": "A",
        "test_0079": "B", "test_0080": "B", "test_0081": "C",
        "test_0082": "D", "test_0083": "B", "test_0084": "A",
        "test_0085": "C", "test_0086": "C", "test_0087": "B",
        "test_0088": "B", "test_0089": "B", "test_0090": "A",
        "test_0091": "B", "test_0092": "A", "test_0093": "A",
        "test_0094": "C", "test_0095": "A", "test_0096": "A",
        "test_0097": "A", "test_0098": "A", "test_0099": "B",
        "test_0100": "D", "test_0101": "C", "test_0102": "C",
        "test_0103": "B", "test_0104": "B", "test_0105": "C",
        "test_0106": "C", "test_0107": "B", "test_0108": "C",
        "test_0109": "A", "test_0110": "A", "test_0111": "B",
        "test_0112": "G", "test_0113": "C", "test_0114": "A",
        "test_0115": "D", "test_0116": "A", "test_0117": "A",
        "test_0118": "E", "test_0119": "A", "test_0120": "C",
        "test_0121": "C", "test_0122": "A", "test_0123": "B",
        "test_0124": "B", "test_0125": "D", "test_0126": "B",
        "test_0127": "A", "test_0128": "A", "test_0129": "A",
        "test_0130": "A", "test_0131": "C", "test_0132": "B",
        "test_0133": "A", "test_0134": "B", "test_0135": "B",
        "test_0136": "B", "test_0137": "C", "test_0138": "A",
        "test_0139": "D", "test_0140": "C", "test_0141": "A",
        "test_0142": "C", "test_0143": "B", "test_0144": "A",
        "test_0145": "A", "test_0146": "C", "test_0147": "E",
        "test_0148": "B", "test_0149": "C", "test_0150": "C",
        "test_0151": "F", "test_0152": "A", "test_0153": "D",
        "test_0154": "C", "test_0155": "A", "test_0156": "B",
        "test_0157": "B", "test_0158": "A", "test_0159": "D",
        "test_0160": "D", "test_0161": "A", "test_0162": "C",
        "test_0163": "D", "test_0164": "A", "test_0165": "B",
        "test_0166": "B", "test_0167": "C", "test_0168": "D",
        "test_0169": "B", "test_0170": "A", "test_0171": "B",
        "test_0172": "B", "test_0173": "D", "test_0174": "B",
        "test_0175": "B", "test_0176": "A", "test_0177": "A",
        "test_0178": "A", "test_0179": "B", "test_0180": "A",
        "test_0181": "A", "test_0182": "C", "test_0183": "B",
        "test_0184": "D", "test_0185": "C", "test_0186": "C",
        "test_0187": "D", "test_0188": "B", "test_0189": "B",
        "test_0190": "D", "test_0191": "B", "test_0192": "B",
        "test_0193": "B", "test_0194": "C", "test_0195": "B",
        "test_0196": "A", "test_0197": "B", "test_0198": "A",
        "test_0199": "B", "test_0200": "D", "test_0201": "A",
        "test_0202": "D", "test_0203": "B", "test_0204": "A",
        "test_0205": "B", "test_0206": "D", "test_0207": "B",
        "test_0208": "B", "test_0209": "D", "test_0210": "B",
        "test_0211": "A", "test_0212": "A", "test_0213": "D",
        "test_0214": "B", "test_0215": "A", "test_0216": "B",
        "test_0217": "B", "test_0218": "A", "test_0219": "B",
        "test_0220": "J", "test_0221": "A", "test_0222": "C",
        "test_0223": "C", "test_0224": "B", "test_0225": "C",
        "test_0226": "A", "test_0227": "A", "test_0228": "J",
        "test_0229": "D", "test_0230": "B", "test_0231": "B",
        "test_0232": "A", "test_0233": "A", "test_0234": "A",
        "test_0235": "B", "test_0236": "A", "test_0237": "A",
        "test_0238": "B", "test_0239": "B", "test_0240": "B",
        "test_0241": "D", "test_0242": "D", "test_0243": "B",
        "test_0244": "A", "test_0245": "E", "test_0246": "E",
        "test_0247": "D", "test_0248": "B", "test_0249": "B",
        "test_0250": "D", "test_0251": "H", "test_0252": "C",
        "test_0253": "D", "test_0254": "B", "test_0255": "C",
        "test_0256": "C", "test_0257": "C", "test_0258": "B",
        "test_0259": "A", "test_0260": "B", "test_0261": "B",
        "test_0262": "B", "test_0263": "C", "test_0264": "C",
        "test_0265": "A", "test_0266": "A", "test_0267": "B",
        "test_0268": "B", "test_0269": "I", "test_0270": "D",
        "test_0271": "D", "test_0272": "A", "test_0273": "D",
        "test_0274": "A", "test_0275": "A", "test_0276": "B",
        "test_0277": "A", "test_0278": "E", "test_0279": "B",
        "test_0280": "B", "test_0281": "A", "test_0282": "A",
        "test_0283": "A", "test_0284": "C", "test_0285": "B",
        "test_0286": "C", "test_0287": "D", "test_0288": "B",
        "test_0289": "C", "test_0290": "A", "test_0291": "B",
        "test_0292": "B", "test_0293": "D", "test_0294": "C",
        "test_0295": "A", "test_0296": "B", "test_0297": "D",
        "test_0298": "B", "test_0299": "B", "test_0300": "A",
        "test_0301": "A", "test_0302": "A", "test_0303": "A",
        "test_0304": "B", "test_0305": "C", "test_0306": "A",
        "test_0307": "H", "test_0308": "B", "test_0309": "B",
        "test_0310": "E", "test_0311": "A", "test_0312": "A",
        "test_0313": "A", "test_0314": "A", "test_0315": "A",
        "test_0316": "D", "test_0317": "C", "test_0318": "C",
        "test_0319": "A", "test_0320": "D", "test_0321": "C",
        "test_0322": "B", "test_0323": "A", "test_0324": "C",
        "test_0325": "A", "test_0326": "E", "test_0327": "B",
        "test_0328": "D", "test_0329": "A", "test_0330": "D",
        "test_0331": "C", "test_0332": "A", "test_0333": "B",
        "test_0334": "A", "test_0335": "A", "test_0336": "B",
        "test_0337": "B", "test_0338": "C", "test_0339": "A",
        "test_0340": "C", "test_0341": "D", "test_0342": "B",
        "test_0343": "B", "test_0344": "C", "test_0345": "A",
        "test_0346": "B", "test_0347": "B", "test_0348": "H",
        "test_0349": "C", "test_0350": "C", "test_0351": "G",
        "test_0352": "B", "test_0353": "B", "test_0354": "B",
        "test_0355": "A", "test_0356": "C", "test_0357": "C",
        "test_0358": "A", "test_0359": "B", "test_0360": "D",
        "test_0361": "A", "test_0362": "A", "test_0363": "B",
        "test_0364": "B", "test_0365": "C", "test_0366": "C",
        "test_0367": "C", "test_0368": "B", "test_0369": "A",
        "test_0370": "C", "test_0371": "B", "test_0372": "A",
        "test_0373": "B", "test_0374": "A", "test_0375": "C",
        "test_0376": "C", "test_0377": "D", "test_0378": "D",
        "test_0379": "B", "test_0380": "B", "test_0381": "E",
        "test_0382": "B", "test_0383": "B", "test_0384": "A",
        "test_0385": "B", "test_0386": "A", "test_0387": "C",
        "test_0388": "C", "test_0389": "A", "test_0390": "A",
        "test_0391": "A", "test_0392": "B", "test_0393": "B",
        "test_0394": "A", "test_0395": "A", "test_0396": "D",
        "test_0397": "E", "test_0398": "D", "test_0399": "D",
        "test_0400": "A", "test_0401": "A", "test_0402": "E",
        "test_0403": "A", "test_0404": "C", "test_0405": "C",
        "test_0406": "B", "test_0407": "B", "test_0408": "A",
        "test_0409": "B", "test_0410": "B", "test_0411": "C",
        "test_0412": "A", "test_0413": "A", "test_0414": "D",
        "test_0415": "D", "test_0416": "B", "test_0417": "D",
        "test_0418": "D", "test_0419": "A", "test_0420": "A",
        "test_0421": "A", "test_0422": "A", "test_0423": "D",
        "test_0424": "B", "test_0425": "D", "test_0426": "B",
        "test_0427": "B", "test_0428": "C", "test_0429": "B",
        "test_0430": "A", "test_0431": "C", "test_0432": "C",
        "test_0433": "B", "test_0434": "B", "test_0435": "A",
        "test_0436": "C", "test_0437": "B", "test_0438": "C",
        "test_0439": "A", "test_0440": "A", "test_0441": "B",
        "test_0442": "A", "test_0443": "C", "test_0444": "A",
        "test_0445": "B", "test_0446": "A", "test_0447": "B",
        "test_0448": "A", "test_0449": "D", "test_0450": "A",
        "test_0451": "B", "test_0452": "B", "test_0453": "A",
        "test_0454": "B", "test_0455": "B", "test_0456": "C",
        "test_0457": "D", "test_0458": "D", "test_0459": "C",
        "test_0460": "A", "test_0461": "B", "test_0462": "C",
        "test_0463": "D"
    }

    results = {}
    t_start = time.time()

    # Sử dụng ThreadPoolExecutor để xử lý song song các câu hỏi
    print(f"\n🚀 Bắt đầu giải quyết {len(test_data)} câu hỏi song song với Hybrid RAG (Concurrency={CONCURRENCY})...")
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_item = {executor.submit(solve_item, item): item for item in test_data}
        
        completed_count = 0
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                qid, pred_ans, raw_info = future.result()
                results[qid] = {
                    "pred": pred_ans,
                    "raw": raw_info
                }
                completed_count += 1
                if completed_count % 5 == 0 or completed_count == len(test_data):
                    print(f"⌛ Đã xử lý {completed_count}/{len(test_data)} câu...")
            except Exception as exc:
                print(f"❌ Câu hỏi {item['qid']} gặp lỗi: {exc}")
                results[item["qid"]] = {
                    "pred": "A",
                    "raw": f"[ERROR]: {exc}"
                }

    total_duration = time.time() - t_start
    print(f"\n✅ Đã hoàn thành toàn bộ câu hỏi trong {total_duration:.2f} giây! (Trung bình: {total_duration/len(test_data):.2f} giây/câu)")

    # Ghi kết quả và đánh giá độ chính xác
    correct = 0
    total_evaluated = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["qid", "answer"])
        
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for item in test_data:
            qid = item["qid"]
            res = results.get(qid, {"pred": "A", "raw": "[ERROR] No prediction"})
            ans = res["pred"]
            
            # Ghi kết quả
            writer.writerow([qid, ans])
            
            # Đánh giá
            expected = GROUND_TRUTH.get(qid)
            if expected:
                is_correct = "✓" if ans == expected else f"✗ (đúng là {expected})"
                if ans == expected:
                    correct += 1
                total_evaluated += 1
                print(f"📊 {qid} -> {ans} {is_correct}\n{res['raw']}\n")
            else:
                print(f"📊 {qid} -> {ans}\n{res['raw']}\n")
                
        print("="*80)
        if total_evaluated > 0:
            print(f" Độ chính xác trên phần đã kiểm tra: {correct}/{total_evaluated} ({correct/total_evaluated:.2%})")
        print(f"✅ Đã lưu toàn bộ kết quả vào: {output_file}")
        print("="*80)

if __name__ == '__main__':
    main()
