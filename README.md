# Hướng dẫn chạy và tái lập kết quả - Đội thi T1S_KMML (Bảng C - Innovator)

Dự án này là giải pháp AI Agent đa tác vụ sử dụng mô hình ngôn ngữ lớn **Qwen3.5-4B-AWQ-4bit** chạy hoàn toàn offline bằng Docker Container để giải quyết tập câu hỏi trắc nghiệm đa lĩnh vực.

Dưới đây là hướng dẫn chi tiết dành cho Ban Giám Khảo để thực thi chấm điểm offline theo đúng chuẩn quy định của BTC.

---

## 1. Cơ chế hoạt động của Container

Container được đóng gói tuân thủ 100% các tiêu chuẩn mô tả trong tài liệu nộp bài Bảng C Innovator của BTC:
1. **Môi trường**: Build từ Base Image hỗ trợ CUDA 12.2 (`nvidia/cuda:12.2.0-devel-ubuntu22.04`).
2. **Thư mục làm việc**: Làm việc tại thư mục `/code` trong Container.
3. **Mã nguồn chạy**: Chạy thông qua lệnh `CMD ["bash", "inference.sh"]` làm entry-point, thực thi `predict.py`.
4. **Đầu vào (Input)**: Tự động tìm kiếm và đọc file kiểm thử tại `/code/private_test.json` (hỗ trợ cả định dạng `.csv` hoặc `.json` khác dưới dạng fallback).
5. **Đầu ra (Output)**: Kết quả được ghi trực tiếp dưới dạng 2 file tại thư mục gốc `/code`:
   - `/code/submission.csv` (Định dạng cột: `qid,answer`)
   - `/code/submission_time.csv` (Định dạng cột: `qid,answer,time` ghi lại thời gian chạy thực tế của từng sample bằng vòng lặp tuần tự).

---

## 2. Luồng xử lý hệ thống (Pipeline Flow)

Hệ thống được thiết kế theo cơ chế **Logit Margin Early Exit** nhằm tối ưu hóa sự cân bằng giữa độ chính xác suy luận (Accuracy) và tốc độ thực thi (Throughput). Sơ đồ luồng xử lý chi tiết như sau:

```text
               [Bắt đầu: Câu hỏi trắc nghiệm]
                             │
                             ▼
              [Bước 1: Chạy Luồng CoT đầu tiên]
                 (Sinh suy luận đầy đủ: T=0.2)
                             │
                             ▼
            [Bước 2: Logit Margin Verification]
           (Sinh 1 token đáp án kèm logprobs: T=0.0)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Margin >= 0.50]                    [Margin < 0.50]
 (Mô hình chọn dứt khoát)          (Mô hình đang phân vân)
            │                                 │
            ▼                                 ▼
     [EARLY EXIT]                     [Bỏ phiếu phụ]
  (Lấy đáp án Luồng 0)            (Kích hoạt Luồng 1, 2)
            │                                 │
            ▼                                 ▼
    [Xuất kết quả] ◄────────────── [Majority Voting]
                                 (Biểu quyết số đông 3 luồng)
```

### Chi tiết luồng xử lý:
1. **Luồng CoT đầu tiên (Luồng 0)**: Mô hình thực hiện Chain-of-Thought suy luận chi tiết ở nhiệt độ thấp (`temperature=0.2`).
2. **Logit Margin Verification**: Hệ thống sinh tiếp 1 token chốt đáp án bằng giải thuật Greedy (`temperature=0.0`) và trích xuất xác suất (`logprobs`) của 5 token hàng đầu.
3. **Tính toán độ phân vân**: 
   * Tính toán hiệu xác suất: $\text{Margin} = P(\text{Top 1}) - P(\text{Top 2})$.
   * Nếu $\text{Margin} \ge 0.50$ (Không phân vân): Kích hoạt cơ chế **Early Exit**, kết xuất ngay đáp án của Luồng 0 nhằm tiết kiệm thời gian (chỉ tốn 1 luồng duy nhất).
   * Nếu $\text{Margin} < 0.50$ (Phân vân): Kích hoạt song song Luồng CoT 1 và Luồng CoT 2, tiến hành biểu quyết đa số (Majority Voting) giữa 3 luồng để tăng độ chính xác tối đa cho câu hỏi khó.

---

## 3. Quy trình xử lý dữ liệu (Data Processing)

Quy trình xử lý dữ liệu của Agent được thực hiện khép kín và tự động hóa cao:
1. **Tự động nhận diện Schema**: Hàm `load_test_data` tự động phát hiện định dạng dữ liệu đầu vào. Hỗ trợ đầy đủ các dạng dữ liệu trắc nghiệm với tên cột đa dạng như `question`, `choices`, `choice_0...`, `option_0...`, hoặc các nhãn lựa chọn cột `A, B, C, D`.
2. **Vietnamese-Aware Regex Extraction**: Để trích xuất đáp án từ văn bản lập luận CoT, hệ thống sử dụng các bộ lọc regex **phân biệt chữ hoa/thường nghiêm ngặt (case-sensitive)** kết hợp với ranh giới từ (`\b[A-J]\b`) và các từ khóa tiếng Việt như `đáp án đúng là`, `chọn`. Kỹ thuật này giúp triệt tiêu hoàn toàn lỗi nhận diện nhầm các từ tiếng Việt tự nhiên viết thường (như *'ngày'*, *'việc'*, *'hành'*) thành ký tự đáp án.
3. **Chuẩn hóa nhãn (Normalization)**: Đối chiếu đáp án chữ cái trích xuất được với danh sách lựa chọn của câu hỏi để đảm bảo tính hợp lệ trước khi ghi kết quả.

---

## 4. Khởi tạo tài nguyên (Resource Initialization)

Hệ thống hoạt động **100% offline**, toàn bộ tài nguyên được đóng gói sẵn trong Container:
1. **Model Weights**: Mô hình `Qwen3.5-4B-AWQ-4bit` (dung lượng 3.8GB) đã được tích hợp trực tiếp vào container tại thư mục `/code/models/Qwen3.5-4B-AWQ-4bit`.
2. **Khởi tạo Offline**: Tokenizer và LLMEngine được khởi tạo với tham số `local_files_only=True` để ngăn chặn bất kỳ hành vi truy cập mạng nào.
3. **Cấu hình VRAM & Engine động**:
   * Hệ thống tự động quét độ dài câu hỏi trong đề thi để cấu hình `max_model_len` tối ưu một cách động, tiết kiệm bộ nhớ đệm KV Cache.
   * Cấu hình `gpu_memory_utilization=0.90` giúp tận dụng 90% bộ nhớ VRAM của GPU.
   * Kích hoạt `enable_prefix_caching=True` của vLLM để lưu trữ và tái sử dụng KV Cache của các phần prompt trùng lặp giữa các luồng biểu quyết, đẩy throughput lên tối đa.
4. **Cơ chế chống tràn ngữ cảnh chủ động (Proactive Context Overflow Prevention)**:
   * Tích hợp kiểm tra đệ quy thời gian thực (`tokenizer.encode`) trước khi gửi request Bước 2. Nếu phát hiện nguy cơ tràn ngữ cảnh (do ranh giới từ bị nở token khi decode/encode), hệ thống sẽ đệ quy cắt tỉa phần suy luận (reasoning).
   * Hỗ trợ cơ chế cứu vây dự phòng (Fallback Extraction) trích xuất trực tiếp đáp án bằng regex từ bài làm Bước 1 nếu câu hỏi gốc quá dài, giúp bảo toàn 100% ngữ cảnh câu hỏi và tuyệt đối không làm crash container.

---

## 5. Hướng dẫn chạy cho Ban Giám Khảo

### Bước 1: Kéo Image từ Docker Hub
```bash
docker pull khanhle1406/t1s_kmml_innovator:latest
```

### Bước 2: Thực thi chấm điểm tự động (Theo chuẩn BTC)
Hệ thống chấm điểm của BTC sẽ chạy container bằng cách mount file dữ liệu kiểm thử vào `/code/private_test.json`:
```bash
docker run --gpus all \
  -v /path/to/private_test.json:/code/private_test.json \
  khanhle1406/t1s_kmml_innovator:latest
```
Sau khi chạy xong, 2 file kết quả `submission.csv` và `submission_time.csv` sẽ được sinh ra ở thư mục làm việc `/code/`.

### Bước 3: Chạy thử nghiệm Local (Tương thích ngược)
Nếu muốn chạy thử nghiệm cục bộ bằng cách ánh xạ thư mục dữ liệu và thư mục kết quả:
```bash
mkdir -p ./data_mock ./output_mock
# Đặt file test mẫu vào ./data_mock (ví dụ: public_test.json)
docker run --gpus all \
  -v "$(pwd)/data_mock:/data" \
  -v "$(pwd)/output_mock:/output" \
  khanhle1406/t1s_kmml_innovator:latest
```
Hệ thống sẽ tự nhận diện dữ liệu trong `/data` và ghi kết quả bản sao ra `./output_mock/submission.csv` và `./output_mock/submission_time.csv`.

---

## 6. Cấu trúc mã nguồn trong Repository
* **Dockerfile**: Đóng gói môi trường CUDA 12.2 và vLLM offline.
* **requirements.txt**: Danh sách các thư viện và phiên bản chi tiết.
* **inference.sh**: Script khởi chạy chính.
* **predict.py**: Script giải đề thi chính (đọc dữ liệu, chạy vLLM batching cho vote, ghi nhận thời gian chạy của từng sample theo loop tuần tự).
* **NOP_BAI_T1S_KMML/**: Thư mục chứa báo cáo thuyết minh giải pháp gửi Ban Giám Khảo (`T1S_KMML_Thuyet_minh_Innovator.docx`, `T1S_KMML_Thuyet_minh_Innovator.pdf`).
