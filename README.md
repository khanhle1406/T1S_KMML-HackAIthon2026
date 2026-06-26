# Vietnamese Student HackAIthon 2026 - Bảng C (Innovator)

Dự án này là giải pháp AI Agent đa tác vụ sử dụng mô hình ngôn ngữ lớn **Qwen3.5-4B-AWQ-4bit** được tối ưu hóa hiệu năng và độ chính xác để giải quyết tập câu hỏi trắc nghiệm đa ngành. Hệ thống được đóng gói hoàn chỉnh bằng Docker và chạy hoàn toàn offline bằng `LLMEngine` của vLLM.

---

## 1. Yêu cầu nộp bài Bảng C (Innovator)

Theo thể lệ cuộc thi, các tệp tin và thông tin bắt buộc phải chuẩn bị để nộp bao gồm:

1. **Docker Image (Đẩy lên Docker Hub)**: Đường dẫn Docker Hub chứa image đã được build.
2. **Mã nguồn (GitHub Repository)**: Chứa toàn bộ mã nguồn, cấu hình Docker và hướng dẫn chạy tái lập kết quả.
3. **Tài liệu thuyết minh phương pháp**: Tệp tin thuyết minh các chiến lược tối ưu mô hình, tham khảo tại [K1S_KMML_Thuyet_minh_Innovator.docx](file:///Users/xuannguyen/Desktop/Hackaithon/K1S_KMML_Thuyet_minh_Innovator.docx).

---

## 2. Hướng dẫn đóng gói và chạy bằng Docker

### 2.1. Chuẩn bị thư mục dữ liệu
Tạo thư mục dữ liệu kiểm thử trên máy chủ (hoặc máy cá nhân):
```bash
mkdir -p ./data ./output
```
Đặt tệp câu hỏi trắc nghiệm kiểm thử (định dạng `.csv` hoặc `.json`) vào thư mục `./data` và đặt tên là `public_test.csv` (hoặc `public_test.json` / `private_test.csv`).

### 2.2. Build Docker Image
Chạy lệnh sau tại thư mục gốc của dự án để build image:
```bash
docker build -t hackaithon-solver:latest .
```

### 2.3. Chạy thử nghiệm Container với GPU
Chạy container Docker bằng cách ánh xạ (mount) các thư mục dữ liệu vào container:
```bash
docker run --gpus all \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/output" \
  hackaithon-solver:latest
```

* **Đầu vào**: Tự động phát hiện bất kỳ file dữ liệu nào trong thư mục `/data` (ví dụ: `public_test.csv`, `private_test.json`,...).
* **Đầu ra**: Tự động sinh file kết quả tại thư mục `/output/pred.csv` với cấu trúc 2 cột chuẩn: `qid,answer`.

---

## 3. Hướng dẫn đẩy Image lên Docker Hub để nộp bài

Để nộp đường dẫn Docker Image cho Ban Tổ chức, anh thực hiện các bước sau:

1. **Đăng nhập vào Docker Hub** trên terminal:
   ```bash
   docker login
   ```
2. **Gắn tag** cho Docker image vừa build theo tên tài khoản của anh:
   ```bash
   docker tag hackaithon-solver:latest <username_dockerhub>/hackaithon-solver:latest
   ```
   *(Thay `<username_dockerhub>` bằng tên tài khoản Docker Hub của anh)*
3. **Đẩy image lên Docker Hub**:
   ```bash
   docker push <username_dockerhub>/hackaithon-solver:latest
   ```
4. **Sao chép đường dẫn** dạng `<username_dockerhub>/hackaithon-solver:latest` để điền vào form nộp bài.

---

## 4. Cấu trúc thư mục dự án

* [entrypoint_solver.py](file:///Users/xuannguyen/Desktop/Hackaithon/entrypoint_solver.py): Script suy luận chính chạy trong Docker container sử dụng `LLMEngine` offline và biểu quyết `N_VOTES=3` (nhiệt độ `0.2`).
* [Dockerfile](file:///Users/xuannguyen/Desktop/Hackaithon/Dockerfile): Cấu hình môi trường Docker đóng gói mô hình và các thư viện cần thiết.
* [K1S_KMML_Thuyet_minh_Innovator.docx](file:///Users/xuannguyen/Desktop/Hackaithon/K1S_KMML_Thuyet_minh_Innovator.docx): Tài liệu thuyết minh phương pháp tối ưu hóa độ chính xác và tốc độ thực thi.
* [solve_test_vllm_offline.py](file:///Users/xuannguyen/Desktop/Hackaithon/solve_test_vllm_offline.py): Script chạy thử nghiệm offline trên máy chủ (chứa bộ dữ liệu Ground Truth để test nhanh độ chính xác).
* [run_docker_test.sh](file:///Users/xuannguyen/Desktop/Hackaithon/run_docker_test.sh): Script shell hỗ trợ chạy nhanh container thử nghiệm.
* `models/Qwen3.5-4B-AWQ-4bit/`: Thư mục chứa weights mô hình gốc để chạy hoàn toàn offline (cần được đặt trong cùng thư mục khi thực hiện build Docker).

---

## 5. Quy trình chạy Client-Server khi phát triển (Tùy chọn)

Nếu muốn phát triển và tinh chỉnh nhanh prompt hoặc logic trích xuất mà không cần chạy lại toàn bộ container Docker:
1. **Khởi động server**:
   ```bash
   ssh <username_server>@<ssh_host> "/home/zeus/content/client_server/manage_server.sh start"
   ```
2. **Đồng bộ code**:
   ```bash
   scp ./client_server/solve_test_vllm_client.py <username_server>@<ssh_host>:/home/zeus/content/client_server/solve_test_vllm_client.py
   ```
3. **Chạy Client suy luận nhanh**:
   ```bash
   ssh <username_server>@<ssh_host> "python3 client_server/solve_test_vllm_client.py"
   ```
4. **Tắt server**:
   ```bash
   ssh <username_server>@<ssh_host> "/home/zeus/content/client_server/manage_server.sh stop"
   ```
