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

def find_input_file(data_dir="/data"):
    """Tìm file kiểm thử đầu vào tại thư mục chỉ định."""
    files = os.listdir(data_dir)
    # Ưu tiên private_test, public_test định dạng csv hoặc json
    for f in ["private_test.csv", "public_test.csv", "private_test.json", "public_test.json"]:
        if f in files:
            return os.path.join(data_dir, f)
    for f in files:
        if f.endswith(".csv") or f.endswith(".json"):
            return os.path.join(data_dir, f)
    raise FileNotFoundError(f"Không tìm thấy file test (.csv hoặc .json) trong {data_dir}!")

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
        input_filepath = find_input_file("/data")
        print(f"📂 Tìm thấy file kiểm thử: {input_filepath}")
        test_data = load_test_data(input_filepath)
        print(f"📊 Đã nạp thành công {len(test_data)} câu hỏi từ file test.")
    except Exception as e:
        print(f"❌ Lỗi đọc dữ liệu đầu vào: {e}")
        # Ghi file rỗng hoặc lỗi và thoát
        os.makedirs("/output", exist_ok=True)
        with open("/output/pred.csv", "w", encoding="utf-8") as f_err:
            f_err.write("qid,answer\n")
        return

    # 2. Nhận diện phần cứng GPU và tối ưu hóa EngineArgs động
    num_gpus = 1
    gpu_mem_util = 0.90
    max_model_len = 4096
    kv_cache_dtype = "auto"
    max_num_seqs = 512
    max_num_batched_tokens = 4096
    
    try:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024 ** 3)
            
            if memory_gb <= 16.5:   # Ví dụ: NVIDIA T4 (16GB VRAM)
                gpu_mem_util = 0.85
                max_model_len = 3072
                max_num_seqs = 256
                max_num_batched_tokens = 2048
            elif memory_gb <= 24.5: # Ví dụ: NVIDIA L4 / A10G (24GB VRAM)
                gpu_mem_util = 0.90
                max_model_len = 4096
                max_num_seqs = 512
                max_num_batched_tokens = 4096
            else:                   # Ví dụ: NVIDIA A100 / H100
                gpu_mem_util = 0.92
                max_model_len = 4096
                max_num_seqs = 1024
                max_num_batched_tokens = 8192
                
            print(f"⚙️ Auto-detected Hardware: {torch.cuda.get_device_name(0)} x{num_gpus} ({memory_gb:.1f} GB VRAM)")
            
            # Tự động chọn KV Cache (dùng auto để tương thích với FLASH_ATTN và đạt độ chính xác cao nhất)
            device_capability = torch.cuda.get_device_capability(0)
            kv_cache_dtype = "auto"
            print(f"⚡ GPU Compute Capability {device_capability[0]}.{device_capability[1]}. Dùng auto KV Cache (bằng kiểu dữ liệu mô hình để tối ưu hóa độ chính xác).")
            print(f"⚙️ Auto-configured: gpu_memory_utilization={gpu_mem_util}, max_model_len={max_model_len}, max_num_seqs={max_num_seqs}")
    except Exception as e:
        print(f"⚠️ GPU auto-config failed: {e}. Sử dụng cấu hình mặc định.")

    # 3. Khởi tạo vLLM Engine
    t_load_start = time.time()
    engine_args = EngineArgs(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=num_gpus,
        enforce_eager=False,          # Bật CUDA Graph
        enable_prefix_caching=True,   # Tiết kiệm prefill giữa các luồng bỏ phiếu
        kv_cache_dtype=kv_cache_dtype,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        attention_backend="FLASH_ATTN"
    )
    engine = LLMEngine.from_engine_args(engine_args)
    print(f"✅ Đã tải xong Model và khởi tạo LLMEngine trong {time.time() - t_load_start:.2f} giây!")

    # 4. Chuẩn bị prompts
    prompts_step1 = []
    for item in test_data:
        choices_text = "\n".join([f"{LETTERS[i]}. {choice}" for i, choice in enumerate(item["choices"])])
        user_prompt = f"Câu hỏi:\n{item['question']}\n\nLựa chọn:\n{choices_text}\n\nNhiệm vụ: Chọn đúng 1 đáp án."
        
        messages = [
            {"role": "system", "content": SYSTEM_REASONING},
            {"role": "user", "content": user_prompt}
        ]
        prompt = apply_chat_template_qwen(messages, add_generation_prompt=True)
        prompt += "Suy luận:\n"
        prompts_step1.append(prompt)

    # 5. Sinh luồng suy luận đồng thời
    t_inference_start = time.time()
    
    # Đẩy toàn bộ câu hỏi vào hàng đợi xử lý song song của engine
    for q_idx, prompt in enumerate(prompts_step1):
        est_tokens = len(prompt) // 3
        max_reasoning_tokens = max(100, 4096 - est_tokens - 50)
        max_reasoning_tokens = min(800, max_reasoning_tokens)
        
        for path_idx in range(N_VOTES):
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

    votes_dict = {i: [None] * N_VOTES for i in range(len(test_data))}
    reasonings_dict = {i: [None] * N_VOTES for i in range(len(test_data))}

    # Vòng lặp sự kiện bất đồng bộ điều phối Step 1 và Step 2
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        
        for request_output in request_outputs:
            if request_output.finished:
                req_id = request_output.request_id
                
                # Xử lý Step 1
                if req_id.startswith("q_"):
                    parts = req_id.split("_")
                    q_idx = int(parts[1])
                    path_idx = int(parts[2])
                    choices_count = len(test_data[q_idx]["choices"])
                    
                    reasoning_text = request_output.outputs[0].text
                    reasonings_dict[q_idx][path_idx] = reasoning_text
                    
                    ans = extract_answer_from_completed(reasoning_text, choices_count)
                    if ans:
                        votes_dict[q_idx][path_idx] = ans
                    else:
                        # Bị cắt cụt -> Đẩy ngay Step 2 vào hàng đợi gối đầu
                        prompt_base = prompts_step1[q_idx] + "\n\nChốt lại, đáp án đúng là chữ cái:"
                        est_base_tokens = len(prompt_base) // 3
                        max_reasoning_budget_tokens = 4096 - est_base_tokens - 10
                        if max_reasoning_budget_tokens < 100:
                            max_reasoning_budget_tokens = 100
                        
                        max_reasoning_budget_chars = max_reasoning_budget_tokens * 3
                        if len(reasoning_text) > max_reasoning_budget_chars:
                            truncated_reasoning = reasoning_text[-max_reasoning_budget_chars:]
                        else:
                            truncated_reasoning = reasoning_text
                            
                        prompt_step2 = prompts_step1[q_idx] + f"{truncated_reasoning}\n\nChốt lại, đáp án đúng là chữ cái:"
                        
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
                
                # Xử lý Step 2 (Fallback)
                elif req_id.startswith("s2_"):
                    parts = req_id.split("_")
                    q_idx = int(parts[1])
                    path_idx = int(parts[2])
                    choices_count = len(test_data[q_idx]["choices"])
                    
                    raw_ans = request_output.outputs[0].text
                    ans = extract_letter(raw_ans, choices_count)
                    ans = normalize_prediction(ans, test_data[q_idx])
                    votes_dict[q_idx][path_idx] = ans

    t_inference_duration = time.time() - t_inference_start
    print(f"✅ Đã hoàn thành suy luận trong {t_inference_duration:.2f} giây! (Trung bình: {t_inference_duration/len(test_data):.2f} giây/câu)")

    # 6. Biểu quyết bầu chọn số đông và xuất file kết quả
    os.makedirs("/output", exist_ok=True)
    output_filepath = "/output/pred.csv"
    
    with open(output_filepath, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["qid", "answer"])
        
        for q_idx, item in enumerate(test_data):
            votes = votes_dict[q_idx]
            # Biểu quyết lấy ý kiến số đông
            vote_counts = Counter(votes)
            selected_ans = vote_counts.most_common(1)[0][0]
            
            writer.writerow([item["qid"], selected_ans])
            
    print(f"🎯 Đã xuất file kết quả thành công vào: {output_filepath}")
    print("====================================================================")
    print("🎉 CONTAINER HOÀN THÀNH NHIỆM VỤ!")
    print("====================================================================")

if __name__ == "__main__":
    main()
