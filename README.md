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

## 2. Hướng dẫn chạy cho Ban Giám Khảo

### Bước 1: Kéo Image từ Docker Hub
Ban Giám Khảo thực hiện kéo Docker Image đã được build sẵn từ Docker Hub về máy chủ:
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

## 3. Cấu trúc mã nguồn trong Repository
* **Dockerfile**: Đóng gói môi trường CUDA 12.2 và vLLM offline.
* **requirements.txt**: Danh sách các thư viện và phiên bản chi tiết.
* **inference.sh**: Script khởi chạy chính.
* **predict.py**: Script giải đề thi chính (đọc dữ liệu, chạy vLLM batching cho vote, ghi nhận thời gian chạy của từng sample theo loop tuần tự).
* **NOP_BAI_T1S_KMML/**: Thư mục chứa báo cáo thuyết minh giải pháp gửi Ban Giám Khảo (`T1S_KMML_Thuyet_minh_Innovator.docx`).
