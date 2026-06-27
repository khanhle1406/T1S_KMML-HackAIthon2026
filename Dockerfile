# BASE IMAGE
# Sử dụng phiên bản CUDA 12.2 và Ubuntu 22.04 để tương thích với Server BTC và có sẵn Python 3.10+
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# SYSTEM DEPENDENCIES
# Cài đặt Python, Pip và các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Link python3 thành python
RUN ln -s /usr/bin/python3 /usr/bin/python

# PROJECT SETUP
# Thiết lập thư mục làm việc theo yêu cầu BTC
WORKDIR /code

# Sao chép toàn bộ source code (và mô hình nếu có) vào trong container
COPY . /code

# Cài đặt các thư viện từ requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Tạo các thư mục data và output và phân quyền ghi (để tương thích ngược với các kiểm thử khác)
RUN mkdir -p /data /output && chmod 777 /data /output

# Khai báo các biến môi trường cần thiết
ENV PYTHONUNBUFFERED=1
ENV VLLM_USE_FLASHINFER_SAMPLER=0
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Chạy bằng user root hoặc có quyền ghi
USER root

# Lệnh chạy mặc định khi container khởi động theo yêu cầu BTC
CMD ["bash", "inference.sh"]
