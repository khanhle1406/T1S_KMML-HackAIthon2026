#!/bin/bash
# Script chạy thử nghiệm container Docker với GPU

# Tạo thư mục data giả lập nếu chưa có
mkdir -p ./data_mock ./output_mock

# Sao chép public-test vào data_mock làm dữ liệu test mẫu
if [ -f "public-test_1780368312.json" ]; then
    cp public-test_1780368312.json ./data_mock/public_test.json
fi

echo "🚀 Đang khởi chạy Docker container với GPU..."
docker run --gpus all \
  -v "$(pwd)/data_mock:/data" \
  -v "$(pwd)/output_mock:/output" \
  hackaithon-solver:latest

echo "✅ Đã chạy xong! Kiểm tra kết quả tại ./output_mock/pred.csv:"
if [ -f "./output_mock/pred.csv" ]; then
    head -n 10 ./output_mock/pred.csv
else
    echo "❌ Lỗi: Không tìm thấy file kết quả tại ./output_mock/pred.csv"
fi
