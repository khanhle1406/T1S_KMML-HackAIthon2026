# Hướng dẫn chạy và tái lập kết quả - Đội thi T1S_KMML (Bảng C - Innovator)

Dự án này là giải pháp AI Agent đa tác vụ sử dụng mô hình ngôn ngữ lớn **Qwen3.5-4B-AWQ-4bit** chạy hoàn toàn offline bằng Docker Container để giải quyết tập câu hỏi trắc nghiệm đa lĩnh vực.

Dưới đây là hướng dẫn chi tiết dành cho Ban Giám Khảo để khởi động và thực thi chấm điểm offline.

---

## 1. Hướng dẫn chạy bằng Docker (Khuyên dùng)

### Bước 1: Kéo Image từ Docker Hub
Ban Giám Khảo thực hiện kéo Docker Image đã được build sẵn từ Docker Hub về máy chủ:
```bash
docker pull khanhle1406/t1s_kmml_innovator:latest
```

### Bước 2: Chuẩn bị thư mục dữ liệu kiểm thử
Tạo các thư mục chứa tệp tin câu hỏi kiểm thử và tệp tin kết quả trên máy chủ:
```bash
mkdir -p ./data ./output
```
* Đặt tệp tin câu hỏi kiểm thử dạng CSV hoặc JSON (ví dụ: `public_test.json` hoặc `private_test.csv`) vào trong thư mục `./data`.
* Hệ thống của chúng tôi được thiết kế linh hoạt, tự động nhận diện và đọc bất kỳ tệp tin `.csv` hoặc `.json` nào có trong thư mục `/data`.

### Bước 3: Thực thi Docker Container với GPU
Chạy lệnh sau để ánh xạ thư mục dữ liệu vào container và bắt đầu quá trình suy luận offline:
```bash
docker run --gpus all \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/output" \
  khanhle1406/t1s_kmml_innovator:latest
```

* **Đầu vào (Input)**: Đọc từ thư mục mount `/data` (ví dụ: `/data/private_test.csv`).
* **Đầu ra (Output)**: Kết quả dự đoán được ghi tự động vào `/output/pred.csv` với cấu trúc chuẩn gồm 2 cột: `qid,answer`.

---

## 2. Cấu trúc mã nguồn trong Repository
* **Dockerfile**: Đóng gói môi trường vLLM offline và các thư viện hỗ trợ.
* **entrypoint_solver.py**: Script điều phối suy luận bất đồng bộ gối đầu, tự động bóc tách regex phân biệt hoa-thường và biểu quyết tự đồng thuận (`N_VOTES=3`).
* **T1S_KMML_Thuyet_minh_Innovator.docx**: Tài liệu thuyết minh giải pháp gửi Ban Giám Khảo.
