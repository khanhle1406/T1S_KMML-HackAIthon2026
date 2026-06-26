FROM vllm/vllm-openai:v0.6.3

# Nâng cấp thư viện transformers và vllm để đồng bộ với môi trường chạy và hỗ trợ kiến trúc mô hình qwen3_5
RUN pip install --no-cache-dir vllm==0.23.0 transformers==5.12.1

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép weights của mô hình từ thư mục cục bộ vào container để chạy hoàn toàn offline
COPY models/Qwen3.5-4B-AWQ-4bit /app/models/Qwen3.5-4B-AWQ-4bit

# Sao chép script giải đề thi vào container
COPY entrypoint_solver.py /app/entrypoint_solver.py

# Đảm bảo các thư mục /data và /output tồn tại trong container
RUN mkdir -p /data /output

# Chạy bằng user root hoặc có quyền ghi vào /output
USER root

# Khai báo biến môi trường cho PyTorch và vLLM
ENV PYTHONUNBUFFERED=1
ENV VLLM_USE_FLASHINFER_SAMPLER=0
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Thiết lập entrypoint chạy trực tiếp script giải đề thi khi container khởi động
ENTRYPOINT ["python3", "/app/entrypoint_solver.py"]
