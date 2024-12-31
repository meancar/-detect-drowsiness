import cv2
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input 
import pygame
from ultralytics import YOLO
import time
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1) # Load model từ file .h5

model_resnet = tf.keras.models.load_model('models\\resnet50pro.h5')

# Load YOLO model
model_yolo = YOLO("models\\YOLO.pt")

# Chuẩn bị webcam
cap = cv2.VideoCapture(0)

# Khởi tạo mixer của pygame
pygame.mixer.init()

# Load âm thanh cảnh báo
alert_sound = pygame.mixer.Sound("sound (mp3cut.net).mp3")

# Khởi tạo biến đếm và ngưỡng
count_sleepy = 0
count_normal = 0
threshold = 2  # Ngưỡng số lần phát hiện "Sleepy" liên tục trong 3 giây

# Thời điểm bắt đầu tính số lần "Sleepy" liên tục
start_time = time.time()

while True:
    # Chụp một khung hình từ webcam
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Đảo ngược hình ảnh

    # Dự đoán bounding boxes sử dụng YOLO
    results = model_yolo.predict(frame)
    print(results)  # Debug output
    
    # Trích xuất bounding box của khuôn mặt và dự đoán trạng thái
    if len(results) > 0:
        for result in results:
            for box in result.boxes.xyxy:
                # Trích xuất tọa độ của bounding box và chuyển đổi sang integer
                x1, y1, x2, y2 = map(int, box[:4])

                # Cắt khuôn mặt từ ảnh gốc
                face_roi = frame[y1:y2, x1:x2]

                # Tiền xử lý hình ảnh khuôn mặt
                face_resnet = cv2.resize(face_roi, (224, 224))  # Resize hình ảnh về kích thước 224x224
                face_resnet = face_resnet[..., ::-1]  # Chuyển đổi từ BGR sang RGB (OpenCV đọc ảnh BGR)
                face_resnet = np.expand_dims(face_resnet, axis=0)  # Thêm một chiều để tạo batch
                face_resnet = preprocess_input(face_resnet)  # Tiền xử lý theo định dạng ResNet50

                # Dự đoán trạng thái sử dụng ResNet50
                predictions = model_resnet.predict(face_resnet)

                # Kiểm tra và xác định trạng thái dựa trên dự đoán
                if predictions[0][0] > predictions[0][1]:
                    trang_thai = "Normal"
                    color = (0, 255, 0)  # Màu xanh lá cho trạng thái "Normal"
                    count_normal += 1
                    count_sleepy = 0  # Đặt lại biến đếm nếu phát hiện trạng thái "Normal"
                else:
                    trang_thai = "Sleepy"
                    color = (0, 0, 255)  # Màu đỏ cho trạng thái "Sleepy"
                    count_sleepy += 1  # Tăng biến đếm nếu phát hiện trạng thái "Sleepy"

    # Kiểm tra thời gian và số lần phát hiện trạng thái trong khoảng thời gian 3 giây
    elapsed_time = time.time() - start_time
    if elapsed_time >= 3:
        if count_sleepy > threshold:
            # Phát âm thanh cảnh báo
            alert_sound.play()
        # Đặt lại thời gian bắt đầu và biến đếm
        start_time = time.time()
        count_sleepy = 0
        count_normal = 0

    # Vẽ hộp giới hạn xung quanh khuôn mặt và hiển thị trạng thái
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, trang_thai, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Hiển thị video real-time
    cv2.imshow('Webcam', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ hiển thị video
cap.release()
cv2.destroyAllWindows()
