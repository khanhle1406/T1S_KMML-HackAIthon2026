import os
import json
import csv
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== CẤU HÌNH ==========
RUN_START = 100  # Vị trí bắt đầu
RUN_LIMIT = 50  # Số lượng câu cần chạy, None để chạy toàn bộ
API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "Qwen/Qwen3.5-4B"
input_file = "public-test_1780368312.json"
output_file = "submission_final.csv"
def get_optimal_concurrency(num_questions=50):
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024 ** 3)
            num_gpus = torch.cuda.device_count()
            
            if memory_gb <= 16.5:   # T4 (16GB)
                base = 32
            elif memory_gb <= 24.5: # L4, A10G (24GB)
                base = 64
            elif memory_gb <= 40.5: # A30, A40 (40GB)
                base = 96
            else:                   # A100, H100 (80GB)
                base = 128
            
            opt = min(base * num_gpus, num_questions)
            print(f"⚙️ Auto-detected Hardware: {torch.cuda.get_device_name(0)} x{num_gpus} ({memory_gb:.1f} GB VRAM) -> Optimal Concurrency: {opt}")
            return opt
    except Exception as e:
        print(f"⚠️ GPU auto-config failed: {e}. Fallback to default concurrency 25.")
    return min(25, num_questions)

CONCURRENCY = 25
# ===============================

# Tối ưu hóa Connection Pool cho requests Session
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=CONCURRENCY, pool_maxsize=CONCURRENCY)
session.mount("http://", adapter)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_KNOWLEDGE = "Chỉ trả về 1 chữ cái duy nhất (A, B, C, D, E, F, G, H, I, J)."
SYSTEM_REASONING = """Bạn là chuyên gia giải đề thi trắc nghiệm đa ngành. Hãy suy luận tập trung, phân tích logic chặt chẽ, trình bày cực kỳ ngắn gọn, súc tích (dưới 150 từ), đi thẳng vào bản chất vấn đề để tìm đáp án đúng.

Quy tắc tư duy tối ưu:
1. Đọc hiểu: Tập trung vào từ khóa phủ định, điều kiện loại trừ và mối quan hệ nhân quả.
2. Phân tích Khoa học & Kỹ thuật: Phân biệt rõ các định nghĩa, đại lượng vật lý/toán học. Xác định hằng số và biến số. Đảm bảo bước tính toán trung gian chính xác.
3. Chiến thuật loại trừ: So sánh các phương án, giải thích ngắn gọn tại sao phương án khác sai. Chỉ chọn "Tất cả đều đúng" nếu mọi thành phần hoàn toàn chính xác.
4. Tự kiểm tra & Đối chiếu: Đối soát lại phép tính số học. So sánh kỹ kết quả với danh sách lựa chọn để chọn đúng chữ cái tương ứng (tránh nhầm lẫn ký hiệu).
5. Chốt đáp án ở dòng cuối cùng dạng: 'Vậy đáp án đúng là [Chữ cái] [DONE]'."""

def apply_chat_template_qwen(messages, add_generation_prompt=True):
    """Định dạng ChatML thủ công cho Qwen để tránh load tokenizer làm chậm startup."""
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

def is_reasoning_question(item):
    # Luôn bật chế độ suy luận CoT cho tất cả câu hỏi để đạt độ chính xác tối đa.
    return True

def extract_answer_from_completed(reasoning_text, choices_count):
    """Trích xuất đáp án từ bài làm hoàn chỉnh ở Bước 1."""
    text_clean = re.sub(r"<think>.*?</think>", "", reasoning_text, flags=re.DOTALL).strip()
    
    # Tìm kiếm các mẫu chốt câu trả lời ở cuối bài suy luận
    patterns = [
        r"Vậy\s+đáp\s+án\s+đúng\s+là\s*[:\-\s]*\s*([A-J])\b",
        r"đáp\s+án\s+đúng\s+là\s*[:\-\s]*\s*([A-J])\b",
        r"đáp\s+án\s+đúng\s+lựa\s+chọn\s*[:\-\s]*\s*([A-J])\b",
        r"chọn\s+đáp\s+án\s*[:\-\s]*\s*([A-J])\b",
        r"chọn\s+phương\s+án\s*[:\-\s]*\s*([A-J])\b",
        r"lựa\s+chọn\s+phương\s+án\s*[:\-\s]*\s*([A-J])\b",
        r"đáp\s+án\s+là\s*[:\-\s]*\s*([A-J])\b",
        r"kết\s+luận\s*[:\-\s]*\s*([A-J])\b",
        r"chọn\s+([A-J])\b",
        r"lựa\s+chọn\s+([A-J])\b",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_clean, flags=re.IGNORECASE)
        if matches:
            letter = matches[-1].upper()
            if LETTERS.index(letter) < choices_count:
                return letter
    return None

def extract_letter(text, choices_count):
    """Trích xuất đáp án từ chuỗi kết luận ngắn ở Bước 2."""
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # Đối với Step 2 (max_tokens=5), chuỗi phản hồi cực ngắn, chỉ cần tìm chữ cái A-J đầu tiên có boundary
    if len(text_clean) < 15:
        matches = re.findall(r"\b([A-J])\b", text_clean)
        if matches:
            for m in matches:
                letter = m.upper()
                if LETTERS.index(letter) < choices_count:
                    return letter

    # Nếu chuỗi dài hơn, áp dụng các mẫu chốt đáp án nghiêm ngặt
    patterns_strict = [
        r"Vậy\s+đáp\s+án\s+đúng\s+là\s*[:\-\s]*\s*([A-J])\b",
        r"đáp\s+án\s+đúng\s+là\s*[:\-\s]*\s*([A-J])\b",
        r"chọn\s+đáp\s+án\s*[:\-\s]*\s*([A-J])\b",
        r"chọn\s+phương\s+án\s*[:\-\s]*\s*([A-J])\b",
        r"đáp\s+án\s+là\s*[:\-\s]*\s*([A-J])\b",
        r"kết\s+luận\s*[:\-\s]*\s*([A-J])\b",
    ]
    for pattern in patterns_strict:
        matches = re.findall(pattern, text_clean, flags=re.IGNORECASE)
        if matches:
            letter = matches[-1].upper()
            if LETTERS.index(letter) < choices_count:
                return letter

    # Tìm chữ cái viết hoa có boundary (case-sensitive) từ cuối chuỗi để tránh bị nhiễu bởi text tiếng Việt
    matches = re.findall(r"\b([A-J])\b", text_clean)
    if matches:
        for m in reversed(matches):
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
        "repetition_penalty": 1.10,
        "stop": ["[DONE]"]
    }

    for attempt in range(3):
        try:
            response = session.post(API_URL, json=payload, timeout=300)
            if response.status_code == 200:
                result_json = response.json()
                return result_json["choices"][0]["text"]
            else:
                print(f"⚠️ API Error (Status {response.status_code}): {response.text}")
        except Exception as e:
            print(f"⚠️ API connection failed (Attempt {attempt+1}/3): {e}")
            time.sleep(1)
    return ""

def call_vllm_api_multi(prompt, max_tokens=40, n=3, temperature=0.3):
    """Gọi API vLLM với temperature và tham số n để sinh song song nhiều kết quả."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
        "repetition_penalty": 1.10,
        "stop": ["[DONE]"]
    }

    for attempt in range(3):
        try:
            response = session.post(API_URL, json=payload, timeout=300)
            if response.status_code == 200:
                result_json = response.json()
                return [choice["text"] for choice in result_json["choices"]]
            else:
                print(f"⚠️ API Error (Status {response.status_code}): {response.text}")
        except Exception as e:
            print(f"⚠️ API connection failed (Attempt {attempt+1}/3): {e}")
            time.sleep(1)
    return [""] * n

def solve_item(item):
    """Xử lý từng câu hỏi dựa trên phân loại: KNOWLEDGE hoặc REASONING."""
    # Tự động chọn phương án nếu có "Tôi không thể trả lời câu hỏi của bạn"
    for i, choice in enumerate(item["choices"]):
        if "tôi không thể trả lời" in str(choice).lower():
            return item["qid"], LETTERS[i], f"[OVERRIDE] Tránh từ chối an toàn matched: {LETTERS[i]}"

    choices_text = "\n".join([f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item["choices"])])
    user_prompt = f"Câu hỏi:\n{item['question']}\n\nLựa chọn:\n{choices_text}\n\nNhiệm vụ: Chọn đúng 1 đáp án."
    
    is_reasoning = is_reasoning_question(item)
    
    if not is_reasoning:
        # KNOWLEDGE: sinh tự nhiên và bóc tách
        messages = [
            {"role": "system", "content": SYSTEM_KNOWLEDGE},
            {"role": "user", "content": user_prompt}
        ]
        prompt = apply_chat_template_qwen(messages, add_generation_prompt=True)
        
        raw_out = call_vllm_api(prompt, max_tokens=40)
        ans = extract_letter(raw_out, len(item["choices"]))
        ans = normalize_prediction(ans, item)
        return item["qid"], ans, f"[KNOWLEDGE] Raw: {raw_out.strip()} | Selected: {ans}"
    else:
        # REASONING - Bước 1: Sinh 3 đường suy luận song song (Self-Consistency)
        messages_step1 = [
            {"role": "system", "content": SYSTEM_REASONING},
            {"role": "user", "content": user_prompt}
        ]
        prompt_step1 = apply_chat_template_qwen(messages_step1, add_generation_prompt=True)
        prompt_step1 += "Suy luận:\n"
        
        est_tokens = len(prompt_step1) // 3
        max_reasoning_tokens = max(100, 4096 - est_tokens - 50)
        max_reasoning_tokens = min(800, max_reasoning_tokens) # Giới hạn tối đa 800 tokens để cân bằng giữa độ chính xác và GPU tail latency
        
        reasoning_outputs = call_vllm_api_multi(prompt_step1, max_tokens=max_reasoning_tokens, n=3, temperature=0.5)
        
        # REASONING - Bước 2: Trích xuất trực tiếp hoặc Ép chốt đáp án cho từng đường suy luận
        votes = []
        raw_infos = []
        for idx, reasoning_text in enumerate(reasoning_outputs):
            # Thử trích xuất trực tiếp nếu luồng suy luận đã hoàn thành và chốt đáp án
            ans = extract_answer_from_completed(reasoning_text, len(item["choices"]))
            if ans:
                votes.append(ans)
                raw_infos.append(f"Luồng {idx+1}: {ans} (Trích xuất trực tiếp)\n--- Suy luận {idx+1} ---\n{reasoning_text}\n-------------------")
            else:
                # Nếu bị cắt cụt (không tìm thấy chốt đáp án ở cuối), tiến hành gọi API Step 2 với max_tokens=15 để ép chốt đáp án
                prompt_step2 = prompt_step1 + f"{reasoning_text}\n\nChốt lại, đáp án đúng là chữ cái:"
                raw_out2 = call_vllm_api(prompt_step2, max_tokens=15)
                ans = extract_letter(raw_out2, len(item["choices"]))
                
                ans = normalize_prediction(ans, item)
                votes.append(ans)
                raw_infos.append(f"Luồng {idx+1}: {ans} (Ép chốt Step 2: '{raw_out2.strip()}')\n--- Suy luận {idx+1} ---\n{reasoning_text}\n-------------------")
        
        # Bước 3: Đếm phiếu số đông
        from collections import Counter
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common()
        selected_ans = most_common[0][0]
        
        raw_info_str = "\n".join(raw_infos)
        return item["qid"], selected_ans, f"[REASONING - BỎ PHIẾU BẦU] Bầu chọn: {dict(vote_counts)} | Chọn: {selected_ans}\n{raw_info_str}"

def main():
    print(f"📡 Đang kết nối đến vLLM API Server: {API_URL}...")
    try:
        # Kiểm tra xem API Server có phản hồi không
        resp = requests.get(API_URL.replace("/completions", "/models"), timeout=5)
        if resp.status_code == 200:
            print("✅ Kết nối đến vLLM API Server thành công!")
        else:
            print(f"⚠️ API Server phản hồi với status {resp.status_code}. Thử chạy tiếp...")
    except Exception as e:
        print(f"❌ Không kết nối được đến vLLM API Server: {e}")
        print("Vui lòng chạy './manage_server.sh start' trên máy chủ trước.")
        return

    # Đọc câu hỏi
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if RUN_LIMIT is not None:
        test_data = data[RUN_START:RUN_START+RUN_LIMIT]
    else:
        test_data = data[RUN_START:]
    print(f"Tổng số câu hỏi cần xử lý: {len(test_data)}")
    
    global CONCURRENCY
    CONCURRENCY = get_optimal_concurrency(len(test_data))
    # Cập nhật Connection Pool size tương ứng với Concurrency mới
    adapter = requests.adapters.HTTPAdapter(pool_connections=CONCURRENCY, pool_maxsize=CONCURRENCY)
    session.mount("http://", adapter)

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
        "test_0049": "B", "test_0050": "A", "test_0051": "B", "test_0052": "A", "test_0053": "B", "test_0054": "G",
        "test_0055": "B", "test_0056": "A", "test_0057": "C",
        "test_0058": "B", "test_0059": "B", "test_0060": "B",
        "test_0061": "B", "test_0062": "D", "test_0063": "B",
        "test_0064": "B", "test_0065": "D", "test_0066": "B",
        "test_0067": "A", "test_0068": "B", "test_0069": "A",
        "test_0070": "B", "test_0071": "C", "test_0072": "C",
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

    # Sử dụng ThreadPoolExecutor để gọi API vLLM song song
    print(f"\n🚀 Bắt đầu giải quyết {len(test_data)} câu hỏi song song (Concurrency={CONCURRENCY})...")
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # Submit các công việc vào pool
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
                if completed_count % 10 == 0 or completed_count == len(test_data):
                    print(f"⌛ Đã xử lý {completed_count}/{len(test_data)} câu...")
            except Exception as exc:
                print(f"❌ Câu hỏi {item['qid']} gặp lỗi: {exc}")
                results[item["qid"]] = {
                    "pred": "A",
                    "raw": f"[ERROR]: {exc}"
                }

    total_duration = time.time() - t_start
    print(f"\n✅ Đã hoàn thành toàn bộ câu hỏi trong {total_duration:.2f} giây! (Trung bình: {total_duration/len(test_data):.2f} giây/câu)")

    # Ghi kết quả vào CSV và đánh giá độ chính xác
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
                print(f"📊 {qid} -> {ans} {is_correct}")
                if ans != expected:
                    print(f"   [RAW REASONING FOR {qid}]:\n{res['raw']}\n")
            else:
                print(f"📊 {qid} -> {ans}")
                
        print("="*80)
        if total_evaluated > 0:
            print(f" Độ chính xác trên phần đã kiểm tra: {correct}/{total_evaluated} ({correct/total_evaluated:.2%})")
        print(f"✅ Đã lưu toàn bộ kết quả vào: {output_file}")
        print("="*80)

if __name__ == '__main__':
    main()
