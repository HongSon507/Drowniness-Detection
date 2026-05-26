# Hệ Thống Phát Hiện Buồn Ngủ (Drowsiness Detection)

Dự án này là hệ thống nhận diện trạng thái buồn ngủ của tài xế theo thời gian thực qua webcam, dựa trên kiến trúc **MobileNetV2** và **MediaPipe Face Mesh**, áp dụng thuật toán tối ưu hoá từ bài báo NITYMED (2023)

- Link tham khảo : https://www.mdpi.com/2076-3417/13/13/7849

## Tính năng chính
- Nhận diện khuôn mặt và trích xuất toạ độ 2 mắt độc lập bằng **MediaPipe Face Mesh**.
- Phân tích trạng thái mắt (Mở/Nhắm) bằng mạng CNN hạng nhẹ **MobileNetV2** được tối ưu hóa.
- Tự động cảnh báo buồn ngủ nếu phát hiện mắt nhắm liên tục quá **300ms** (Áp dụng logic theo bài báo docs).
- Cảnh báo "Face Lost" ngay lập tức nếu khuôn mặt lọt khỏi tầm nhìn của Camera.
- Dùng Early Stopping thay vì accuracy phù hợp cho dataset lệch lớp .
- **Tối ưu hóa cho CPU**: Sử dụng cơ chế Đa luồng (Multi-threading) để tách biệt luồng xử lý AI và luồng hiển thị Webcam, giúp tốc độ mượt mà không bị giật lag 

## Cấu trúc thư mục
- `docs/`: Chứa tài liệu nghiên cứu tham khảo.
- `results/`: Chứa file Model đã huấn luyện (`best_model.pth`) và các biểu đồ phân tích.
- `test_images/`: Chứa các ảnh mẫu dùng để test.
- `predict.py`: File chạy chính. Bật Webcam để nhận diện trực tiếp.
- `test_single_image.py`: File kiểm tra trạng thái trên một bức ảnh tĩnh.
- `train.py` & `preprocess.py`: Các script để xử lý dữ liệu và huấn luyện lại Model.
- `data/`: Chứa dataset ảnh mắt (train/val/test, awake/sleepy).

## Dataset
Dataset lấy từ **MRL Infrared Eye Images Dataset for Drowsiness Detection (Forked Version)**.
- Tải file `data.zip` tại trang **Releases**: https://github.com/HongSon507/Drowsiness-Detection/releases/tag/v1.0-data

Sau khi tải về, giải nén vào thư mục gốc:
```
unzip data.zip -d data/
```
Cấu trúc sau khi giải nén:
```
data/
├── train/awake/
├── train/sleepy/   
├── val/awake/
├── val/sleepy/
├── test/awake/
└── test/sleepy/
```

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Anaconda/Miniconda. Sau đó chạy các lệnh sau:
```
conda create -n nd python=3.10
conda activate nd
pip install -r requirements.txt
```

### 2. Chạy Webcam thời gian thực
Kích hoạt môi trường và chạy file `predict.py`:
```
python predict.py
```

### 3. Test trên 1 ảnh tĩnh
Bạn có thể test trực tiếp bằng cách truyền đường dẫn ảnh vào cmd với lệnh:
```
python test_single_image.py test_images/buonngu.jpg
```
### 4. Credits
- HongSon507 inspired from NITYMED Paper (2023)
- Download free dataset from MRL Infrared Eye Images Dataset for Drowsiness Detection (Forked Version)
### 5. Các hướng có thể cải tiến 
- Cải thiện logic cảnh báo thay vì từng frame 
- Có thể thêm các đặc trưng bổ sung để tăng độ chính xác 
(góc nghiêng đầu ,...)
- sử dụng các kiến trúc xử lý ảnh khác thay vì CNN như transformers Vision
# License
MIT License

