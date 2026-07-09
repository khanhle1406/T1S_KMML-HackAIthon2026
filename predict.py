import os
import json
import csv
import re
import time
import ast
from collections import Counter
import torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind

# ========== CẤU HÌNH MẶC ĐỊNH ==========
MODEL_PATH = "/code/models/Qwen3.5-4B-AWQ-4bit"
if not os.path.exists(MODEL_PATH):
    # Fallback cho các môi trường chạy khác
    MODEL_PATH = "/app/models/Qwen3.5-4B-AWQ-4bit"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_REASONING = """Bạn là chuyên gia giải đề thi trắc nghiệm đa ngành. Hãy suy luận từng bước một cách cẩn thận nhưng ngắn gọn (dưới 250 từ), đi thẳng vào bản chất vấn đề để tìm đáp án đúng.

Quy tắc tư duy tối ưu:
1. Đọc hiểu chi tiết: Tập trung vào các từ khóa phủ định, điều kiện loại trừ và bối cảnh được cung cấp. Không giả định thêm các thông tin ngoài văn bản.
2. Đọc hiểu văn bản (Đoạn thông tin): Ưu tiên các thông tin được viết trực tiếp, rõ ràng trong văn bản. Không cố gắng suy diễn sâu xa, phân tích quá mức hoặc bỏ qua các liên kết nhân quả trực tiếp được nêu trong đoạn văn.
3. Suy luận từng bước (Chain-of-Thought): Trình bày các bước lập luận rõ ràng, viết ngắn gọn. Đối với câu hỏi tính toán, hãy ghi rõ công thức sử dụng, thay thế số liệu và kiểm tra kết quả của từng bước tính toán trung gian.
4. Chiến thuật loại trừ: So sánh kỹ các phương án lựa chọn, lý giải vì sao các phương án khác chưa chính xác.
5. Cực kỳ cẩn thận trong việc đối chiếu đáp án chữ cái: Sau khi tính toán ra kết quả hoặc lý luận ra phương án, hãy nhìn thật kỹ danh sách lựa chọn để chọn đúng chữ cái tương ứng (ví dụ: nếu kết quả tính ra là một con số hoặc đáp án cụ thể, hãy đối chiếu xem nó nằm ở lựa chọn nào: A, B, C, D, E... và chốt đúng chữ cái đó).
6. Chốt đáp án ở dòng cuối cùng dạng: 'Vậy đáp án đúng là [Chữ cái] [DONE]'."""

# 3 luồng biểu quyết cùng nhiệt độ 0.2 theo tối ưu của thí sinh
N_VOTES = 3
TEMPS = [0.2, 0.2, 0.2]
# =======================================

def apply_chat_template_qwen(messages, add_generation_prompt=True):
    """Định dạng ChatML thủ công cho Qwen để tránh load tokenizer."""
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

def extract_answer_from_completed(reasoning_text, choices_count):
    """Trích xuất đáp án trực tiếp từ bài làm hoàn chỉnh ở Bước 1."""
    text_clean = re.sub(r"<think>.*?</think>", "", reasoning_text, flags=re.DOTALL).strip()
    
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
    
    if len(text_clean) < 15:
        matches = re.findall(r"\b([A-J])\b", text_clean)
        if matches:
            for m in matches:
                letter = m.upper()
                if LETTERS.index(letter) < choices_count:
                    return letter

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

    matches = re.findall(r"\b([A-J])\b", text_clean)
    if matches:
        for m in reversed(matches):
            letter = m.upper()
            if LETTERS.index(letter) < choices_count:
                return letter
                
    return "A"

def find_input_file():
    """Tìm file kiểm thử đầu vào tại thư mục chỉ định, ưu tiên tuyệt đối /code/private_test.json."""
    # 1. Ưu tiên tuyệt đối file /code/private_test.json theo yêu cầu của BTC
    if os.path.exists("/code/private_test.json"):
        return "/code/private_test.json"
    if os.path.exists("/code/private_test.csv"):
        return "/code/private_test.csv"
        
    # 2. Fallback tìm kiếm trong các thư mục khác để tương thích với test local
    for search_dir in ["/data", "/app/data", "./data", "./data_mock"]:
        if os.path.exists(search_dir):
            try:
                files = os.listdir(search_dir)
                for f in ["private_test.json", "private_test.csv", "public_test.json", "public_test.csv"]:
                    if f in files:
                        return os.path.join(search_dir, f)
                for f in files:
                    if f.endswith(".json") or f.endswith(".csv"):
                        return os.path.join(search_dir, f)
            except Exception:
                pass
    raise FileNotFoundError("Không tìm thấy file test (.json hoặc .csv)!")

def load_test_data(filepath):
    """Đọc dữ liệu cực kỳ linh hoạt (hỗ trợ cả JSON và CSV nhiều định dạng)."""
    data = []
    
    # Kiểm tra nếu là file JSON
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # Đọc định dạng CSV
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("qid") or row.get("id") or ""
            question = row.get("question") or ""
            
            choices = []
            if "choices" in row:
                choices_str = row["choices"].strip()
                try:
                    choices = json.loads(choices_str)
                except Exception:
                    try:
                        choices = ast.literal_eval(choices_str)
                    except Exception:
                        if "\n" in choices_str:
                            choices = [c.strip() for c in choices_str.split("\n") if c.strip()]
                        elif ";" in choices_str:
                            choices = [c.strip() for c in choices_str.split(";") if c.strip()]
                        else:
                            choices = [choices_str]
            else:
                # Quét các cột lựa chọn dạng choice_0, choice_1... hoặc option_0, option_1...
                idx = 0
                while True:
                    found = False
                    for key_pattern in [f"choice_{idx}", f"choice{idx}", f"option_{idx}", f"option{idx}", f"choices_{idx}"]:
                        if key_pattern in row:
                            choices.append(row[key_pattern])
                            found = True
                            break
                    if not found:
                        # Kiểm tra dạng cột A, B, C, D
                        letter = chr(65 + idx)
                        if letter in row:
                            choices.append(row[letter])
                            found = True
                    if not found:
                        break
                    idx += 1
            
            data.append({
                "qid": qid,
                "question": question,
                "choices": choices
            })
    return data

def main():
    print("====================================================================")
    print("🚀 CONTAINER KHỞI ĐỘNG: Đang chạy script giải đề thi HackAIthon 2026...")
    print("====================================================================")
    
    # 1. Tìm và nạp file test
    try:
        input_filepath = find_input_file()
        print(f"📂 Tìm thấy file kiểm thử: {input_filepath}")
        test_data = load_test_data(input_filepath)
        print(f"📊 Đã nạp thành công {len(test_data)} câu hỏi từ file test.")
        run_limit = os.environ.get("RUN_LIMIT")
        if run_limit:
            test_data = test_data[:int(run_limit)]
            print(f"⚠️ Giới hạn số câu chạy thử nghiệm bằng RUN_LIMIT: {len(test_data)} câu.")
    except Exception as e:
        print(f"❌ Lỗi đọc dữ liệu đầu vào: {e}")
        # Ghi file rỗng hoặc lỗi và thoát
        with open("submission.csv", "w", encoding="utf-8") as f_err:
            f_err.write("qid,answer\n")
        with open("submission_time.csv", "w", encoding="utf-8") as f_err_t:
            f_err_t.write("qid,answer,time\n")
        if os.path.exists("/output"):
            with open("/output/submission.csv", "w") as f_err: f_err.write("qid,answer\n")
            with open("/output/pred.csv", "w") as f_err: f_err.write("qid,answer\n")
            with open("/output/submission_time.csv", "w") as f_err_t: f_err_t.write("qid,answer,time\n")
        return

    # 2. Tải Tokenizer trước để quét và thích ứng độ dài ngữ cảnh của đề thi
    from transformers import AutoTokenizer
    print("🚀 Đang tải tokenizer từ model (local files only)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    print("✅ Đã tải xong Tokenizer!")

    print("📊 Đang quét độ dài câu hỏi trong đề thi để tối ưu hóa ngữ cảnh động...")
    max_prompt_tokens = 0
    for item in test_data:
        choices_text = "\n".join([f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item["choices"])])
        user_prompt = f"Câu hỏi:\n{item['question']}\n\nLựa chọn:\n{choices_text}\n\nNhiệm vụ: Chọn đúng 1 đáp án."
        messages = [
            {"role": "system", "content": SYSTEM_REASONING},
            {"role": "user", "content": user_prompt}
        ]
        prompt = apply_chat_template_qwen(messages, add_generation_prompt=True)
        prompt += "Suy luận:\n"
        prompt_len = len(tokenizer.encode(prompt))
        if prompt_len > max_prompt_tokens:
            max_prompt_tokens = prompt_len
            
    print(f"📏 Câu hỏi dài nhất trong đề thi đo được: {max_prompt_tokens} tokens.")
    
    # Cấu hình max_model_len thích ứng động (đảm bảo chứa đủ prompt + 800 token CoT + 50 token dự phòng)
    max_model_len = max_prompt_tokens + 850
    max_model_len = min(32768, max(3072, max_model_len))
    
    # Nếu ngữ cảnh quá lớn (>8192), tắt CUDA Graph để tránh OOM/khởi động lâu lúc compile
    enforce_eager = True if max_model_len > 8192 else False
    # 3. Nhận diện phần cứng GPU và tối ưu hóa EngineArgs động
    num_gpus = 1
    gpu_mem_util = 0.90
    kv_cache_dtype = "auto"
    max_num_seqs = 32
    max_num_batched_tokens = 4096
    
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024 ** 3)
            
            if memory_gb <= 16.5:   # Ví dụ: NVIDIA T4 (16GB VRAM)
                gpu_mem_util = 0.85
                max_num_seqs = 16
                max_num_batched_tokens = 2048
            elif memory_gb <= 24.5: # Ví dụ: NVIDIA L4 / A10G (24GB VRAM)
                gpu_mem_util = 0.90
                max_num_seqs = 16
                max_num_batched_tokens = 4096
            else:                   # Ví dụ: NVIDIA A100 / H100
                gpu_mem_util = 0.92
                max_num_seqs = 32
                max_num_batched_tokens = 8192                
            print(f"⚙️ Auto-detected Hardware: {torch.cuda.get_device_name(0)} x{num_gpus} ({memory_gb:.1f} GB VRAM)")
            
            # Tự động chọn KV Cache (dùng auto để tương thích với FLASH_ATTN và đạt độ chính xác cao nhất)
            device_capability = torch.cuda.get_device_capability(0)
            kv_cache_dtype = "auto"
            print(f"⚡ GPU Compute Capability {device_capability[0]}.{device_capability[1]}. Dùng auto KV Cache.")
            print(f"⚙️ Auto-configured: gpu_memory_utilization={gpu_mem_util}, max_model_len={max_model_len}, max_num_seqs={max_num_seqs}, enforce_eager={enforce_eager}")
    except Exception as e:
        print(f"⚠️ GPU auto-config failed: {e}. Sử dụng cấu hình mặc định.")

    # 4. Khởi tạo vLLM Engine
    t_load_start = time.time()
    engine_args = EngineArgs(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=num_gpus,
        enforce_eager=enforce_eager,
        enable_prefix_caching=True,   # Tiết kiệm prefill giữa các luồng bỏ phiếu
        kv_cache_dtype=kv_cache_dtype,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        attention_backend="FLASH_ATTN"
    )
    engine = LLMEngine.from_engine_args(engine_args)
    print(f"✅ Đã tải xong Model và khởi tạo LLMEngine trong {time.time() - t_load_start:.2f} giây!")

    # 4. Vòng lặp chạy từng câu hỏi một (sequential loop) theo yêu cầu BTC để đo inference time chính xác
    t_inference_start = time.time()
    predictions = []
    import math
    MARGIN_THRESHOLD = 0.50

    for q_idx, item in enumerate(test_data):
        start_time_sample = time.time()
        
        # Chuẩn bị prompt
        choices_text = "\n".join([f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item["choices"])])
        user_prompt = f"Câu hỏi:\n{item['question']}\n\nLựa chọn:\n{choices_text}\n\nNhiệm vụ: Chọn đúng 1 đáp án."
        
        messages = [
            {"role": "system", "content": SYSTEM_REASONING},
            {"role": "user", "content": user_prompt}
        ]
        prompt = apply_chat_template_qwen(messages, add_generation_prompt=True)
        prompt += "Suy luận:\n"
        
        prompt_tokens = len(tokenizer.encode(prompt))
        max_reasoning_tokens = max(50, max_model_len - prompt_tokens - 20)
        max_reasoning_tokens = min(800, max_reasoning_tokens)
        
        # Bước 1: Chỉ đẩy duy nhất Luồng 0 (Path 0 CoT) vào engine
        engine.add_request(
            request_id=f"q_{q_idx}_0",
            prompt=prompt,
            params=SamplingParams(
                temperature=TEMPS[0],
                max_tokens=max_reasoning_tokens,
                repetition_penalty=1.10,
                stop=["[DONE]"],
                output_kind=RequestOutputKind.FINAL_ONLY
            )
        )
            
        votes = [None] * N_VOTES
        reasonings = [None] * N_VOTES
        early_exited = False
        margin_val = 0.0
        
        # Chạy engine cho đến khi hoàn thành các yêu cầu của câu hỏi hiện tại
        while engine.has_unfinished_requests():
            request_outputs = engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    req_id = request_output.request_id
                    choices_count = len(item["choices"])
                    
                    # 1. Nhận kết quả từ Luồng 0
                    if req_id == f"q_{q_idx}_0":
                        reasoning_text = request_output.outputs[0].text
                        reasonings[0] = reasoning_text
                        
                        # Chuẩn bị Bước 2: Sinh 1 token đáp án kèm logprobs để kiểm tra độ phân vân
                        prompt_base = prompt + f"{reasoning_text}\n\nChốt lại, đáp án đúng là chữ cái:"
                        prompt_base_tokens = len(tokenizer.encode(prompt_base))
                        allowed_reasoning_tokens = max_model_len - prompt_base_tokens - 15
                        
                        if allowed_reasoning_tokens > 10:
                            reasoning_tokens = tokenizer.encode(reasoning_text)
                            truncated_reasoning = tokenizer.decode(reasoning_tokens[-allowed_reasoning_tokens:])
                        else:
                            truncated_reasoning = ""
                            
                        prompt_step2 = prompt + f"{truncated_reasoning}\n\nChốt lại, đáp án đúng là chữ cái:"
                        
                        engine.add_request(
                            request_id=f"conf_{q_idx}",
                            prompt=prompt_step2,
                            params=SamplingParams(
                                temperature=0.0,  # Greedy
                                max_tokens=1,
                                logprobs=5,
                                output_kind=RequestOutputKind.FINAL_ONLY
                            )
                        )
                        
                    # 2. Nhận kết quả kiểm tra độ tự tin từ logprobs
                    elif req_id == f"conf_{q_idx}":
                        outputs = request_output.outputs[0]
                        pred_letter = "A"
                        margin_val = 0.0
                        
                        token_probs = {}
                        if outputs.logprobs and len(outputs.logprobs) > 0:
                            logprob_dict = outputs.logprobs[0]
                            for token_id, logprob_obj in logprob_dict.items():
                                token_text = logprob_obj.decoded_token.strip().upper()
                                prob = math.exp(logprob_obj.logprob)
                                if token_text in LETTERS:
                                    token_probs[token_text] = max(token_probs.get(token_text, 0), prob)
                                    
                        if token_probs:
                            sorted_probs = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)
                            pred_letter = sorted_probs[0][0]
                            p1 = sorted_probs[0][1]
                            p2 = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
                            margin_val = p1 - p2
                        else:
                            ans_extracted = extract_answer_from_completed(reasonings[0], choices_count)
                            if ans_extracted:
                                pred_letter = normalize_prediction(ans_extracted, item)
                                margin_val = 0.50
                            else:
                                pred_letter = "A"
                                margin_val = 0.0
                                
                        votes[0] = pred_letter
                        
                        # Quyết định dừng sớm (Early Exit)
                        if margin_val >= MARGIN_THRESHOLD:
                            # Không phân vân -> Chấp nhận đáp án Luồng 0 và kết thúc sớm
                            early_exited = True
                        else:
                            # Phân vân giữa các đáp án -> Kích hoạt Luồng 1 và Luồng 2 để biểu quyết
                            for path_idx in [1, 2]:
                                temp = TEMPS[path_idx]
                                engine.add_request(
                                    request_id=f"q_{q_idx}_{path_idx}",
                                    prompt=prompt,
                                    params=SamplingParams(
                                        temperature=temp,
                                        max_tokens=max_reasoning_tokens,
                                        repetition_penalty=1.10,
                                        stop=["[DONE]"],
                                        output_kind=RequestOutputKind.FINAL_ONLY
                                    )
                                )
                                
                    # 3. Nhận kết quả từ Luồng 1 hoặc Luồng 2 (khi bị kích hoạt thêm)
                    elif req_id.startswith("q_") and (req_id.endswith("_1") or req_id.endswith("_2")):
                        parts = req_id.split("_")
                        path_idx = int(parts[2])
                        
                        reasoning_text = request_output.outputs[0].text
                        reasonings[path_idx] = reasoning_text
                        ans = extract_answer_from_completed(reasoning_text, choices_count)
                        if ans:
                            ans = normalize_prediction(ans, item)
                            votes[path_idx] = ans
                        else:
                            # Step 2 phụ cho luồng 1, 2
                            prompt_base = prompt + f"{reasoning_text}\n\nChốt lại, đáp án đúng là chữ cái:"
                            prompt_base_tokens = len(tokenizer.encode(prompt_base))
                            allowed_reasoning_tokens = max_model_len - prompt_base_tokens - 15
                            
                            if allowed_reasoning_tokens > 10:
                                reasoning_tokens = tokenizer.encode(reasoning_text)
                                truncated_reasoning = tokenizer.decode(reasoning_tokens[-allowed_reasoning_tokens:])
                            else:
                                truncated_reasoning = ""
                                
                            prompt_step2 = prompt + f"{truncated_reasoning}\n\nChốt lại, đáp án đúng là chữ cái:"
                            
                            step2_req_id = f"s2_{q_idx}_{path_idx}"
                            engine.add_request(
                                request_id=step2_req_id,
                                prompt=prompt_step2,
                                params=SamplingParams(
                                    temperature=0.0,
                                    max_tokens=15,
                                    repetition_penalty=1.10,
                                    stop=["[DONE]"],
                                    output_kind=RequestOutputKind.FINAL_ONLY
                                )
                            )
                            
                    # 4. Nhận kết quả Step 2 phụ cho Luồng 1 hoặc Luồng 2
                    elif req_id.startswith("s2_"):
                        parts = req_id.split("_")
                        path_idx = int(parts[2])
                        
                        raw_ans = request_output.outputs[0].text
                        ans = extract_letter(raw_ans, choices_count)
                        ans = normalize_prediction(ans, item)
                        votes[path_idx] = ans
                        
        # Biểu quyết lấy ý kiến số đông
        if early_exited:
            selected_ans = votes[0]
        else:
            valid_votes = [v for v in votes if v is not None]
            vote_counts = Counter(valid_votes)
            selected_ans = vote_counts.most_common(1)[0][0] if valid_votes else "A"
            
        end_time_sample = time.time()
        time_infer_sample = end_time_sample - start_time_sample
        
        # Lưu kết quả
        predictions.append({
            "qid": item["qid"],
            "answer": selected_ans,
            "time": time_infer_sample
        })
        
        if (q_idx + 1) % 10 == 0 or (q_idx + 1) == len(test_data):
            print(f"⏳ Đã xử lý {q_idx + 1}/{len(test_data)} câu. (Thời gian câu vừa rồi: {time_infer_sample:.3f}s)")

    t_inference_duration = time.time() - t_inference_start
    print(f"✅ Đã hoàn thành suy luận trong {t_inference_duration:.2f} giây! (Trung bình: {t_inference_duration/len(test_data):.3f} giây/câu)")

    # 5. Xuất các file kết quả
    # Cấu trúc của BTC yêu cầu: submission.csv, submission_time.csv
    # Chúng tôi sẽ ghi vào cả thư mục hiện tại (WORKDIR /code) và /output (nếu có để tương thích với test local)
    
    output_files = ["submission.csv", "submission_time.csv"]
    for out_f in output_files:
        filepath = out_f
        with open(filepath, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            if out_f == "submission.csv":
                writer.writerow(["qid", "answer"])
                for pred in predictions:
                    writer.writerow([pred["qid"], pred["answer"]])
            else:
                writer.writerow(["qid", "answer", "time"])
                for pred in predictions:
                    writer.writerow([pred["qid"], pred["answer"], f"{pred['time']:.4f}"])
        print(f"🎯 Đã xuất file kết quả: {filepath}")

    # Đồng bộ sang thư mục /output để tương thích ngược với run_docker_test.sh của thí sinh
    if os.path.exists("/output"):
        try:
            import shutil
            shutil.copy("submission.csv", "/output/submission.csv")
            shutil.copy("submission.csv", "/output/pred.csv")
            shutil.copy("submission_time.csv", "/output/submission_time.csv")
            print("🎯 Đã đồng bộ kết quả sang thư mục mount /output")
        except Exception as e:
            print(f"⚠️ Không thể đồng bộ kết quả sang /output: {e}")
            
    print("====================================================================")
    print("🎉 CONTAINER HOÀN THÀNH NHIỆM VỤ!")
    print("====================================================================")

if __name__ == "__main__":
    main()
