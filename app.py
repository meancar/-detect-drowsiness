from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras_facenet import FaceNet  
import time
import pygame
import pickle 
import os
import base64 
from ultralytics import YOLO
import tensorflow as tf

app = Flask(__name__)

# Load FaceNet model
model = FaceNet().model
model_yolo = YOLO("models\\YOLO.pt")
model_resnet = tf.keras.models.load_model('models\\resnet50pro.h5')
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("sound (mp3cut.net).mp3")

# Function to convert image to embedding
def img_to_encoding(image, model):
    image = cv2.resize(image, (160, 160))
    img = np.around(np.array(image) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2) #

# Function to verify identity
def verify(image, database, model):
    # Load database
    try:
        with open('data.pickle', 'rb') as handle:
            database = pickle.load(handle)
    except FileNotFoundError:
        database = {}

    # Update targets list
    targets = list(database.keys())

    encoding = img_to_encoding(image, model)
    min_dist = float('inf')
    min_identity = None
    #Tìm khoảng cách ngắn nhất trong Triplet loss
    for identity in targets:
        dist = np.linalg.norm(encoding - database[identity], ord=2)
        if dist < min_dist:
            min_dist = dist
            min_identity = identity
    if min_identity is not None and min_dist < 0.75:
        print("It's " + str(min_identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not one of the targets, please go away")
        door_open = False
    return min_dist, door_open

# Load database
with open('data.pickle', 'rb') as handle:
    database = pickle.load(handle)

cap = cv2.VideoCapture(0)
targets = list(database.keys())

# Initialize pygame mixer
pygame.mixer.init()

# Load access successful sound
success_sound = pygame.mixer.Sound(r"tingting.mp3")

# Load access denied sound
denied_sound = pygame.mixer.Sound(r"sound (mp3cut.net).mp3")

# Global variables to track access stats
count_successful = 0
count_denied = 0

def generate_frames():
    cap = cv2.VideoCapture(0)

    global count_successful
    global count_denied
    time_limit = 15
    start_time = time.time()
    sound_played = False

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            if not sound_played:
                if count_successful > count_denied:
                    success_sound.play()
                else:
                    denied_sound.play()

                print("Successful accesses:", count_successful)
                print("Denied accesses:", count_denied)

                sound_played = True
                cap.release()
                break

            count_successful = 0
            count_denied = 0
            start_time = time.time()

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results_list = model_yolo(frame)

        if len(results_list) > 0:
            for results in results_list:
                if results is not None and results.boxes is not None:
                    for box in results.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box[:4])

                        face_region = frame[y1:y2, x1:x2]
                        #Trả về khoảng cách, và cờ door_open
                        dist, door_open = verify(face_region, database, model)

                        if door_open:
                            count_successful += 1
                            cv2.putText(frame, "Access Successful", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            count_denied += 0.1
                            cv2.putText(frame, "Access Denied", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #Nén khung hình thành định dạng JPEG.
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        #Gửi khung hình(bounding box) qua HTTP Stream dưới dạng byte (frame_bytes), sử dụng giao thức MIME.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_access_stats')
def get_access_stats():
    return jsonify({'successful_access': count_successful, 'denied_access': count_denied})


def add_person_to_database(database_file, new_person_encoding, new_person_name):
    # Load database
    try:
        with open(database_file, 'rb') as handle:
            database = pickle.load(handle)
    except FileNotFoundError:
        database = {}

    # Add new person's encoding to database
    database[new_person_name] = new_person_encoding

    # Update targets list
    targets.append(new_person_name)

    # Save updated database to file
    with open(database_file, 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("New person added to the database successfully!")

@app.route('/add_person', methods=['POST'])
def add_person():
    # Lấy dữ liệu từ yêu cầu POST
    data = request.json
    new_person_name = data['name']
    image_data = data['image_data']

    # Giải mã hình ảnh từ base64
    image_bytes = image_data.split(',')[1].encode()
    nparr = np.frombuffer(base64.b64decode(image_bytes), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Sử dụng Haar Cascade để phát hiện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({'message': 'No face detected'})

    # Lấy khuôn mặt đầu tiên được phát hiện
    (x, y, w, h) = faces[0]
    face_roi = image[y:y+h, x:x+w]

    # Tạo mã nhúng cho khuôn mặt
    new_person_encoding = img_to_encoding(face_roi, model)

    # Thêm người mới vào cơ sở dữ liệu
    database_file = 'data.pickle'
    add_person_to_database(database_file, new_person_encoding, new_person_name)

    # Trả về thông báo JSON cho người dùng
    return jsonify({'message': 'New person added to the database successfully!'})
@app.route('/register')
def register():
    return render_template('register.html')

def classify_frame(frame):
    alert_flag = False
    results = model_yolo.predict(frame)
    
    if len(results) > 0:
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                face_roi = frame[y1:y2, x1:x2]

                face_resnet = cv2.resize(face_roi, (224, 224))
                face_resnet = face_resnet[..., ::-1] #Đảo ngược màu sắc của ảnh từ BGR (OpenCV mặc định) sang RGB.
                face_resnet = np.expand_dims(face_resnet, axis=0)
                face_resnet = preprocess_input(face_resnet)
                

                predictions = model_resnet.predict(face_resnet)

                if predictions[0][0] > predictions[0][1]:
                    return "Normal", (x1, y1, x2, y2), alert_flag
                else:
                    alert_flag = True
                    return "Sleepy", (x1, y1, x2, y2), alert_flag
                
    return "No face detect", None, denied_sound.play()


def generate_frames_1():
    cap = cv2.VideoCapture(0)
    
    # Initialize variables to count normal and sleepy states
    normal_count = 0
    sleepy_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        state, bbox, alert_flag = classify_frame(frame)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if state == "Normal" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text_color = (0, 255, 0) if state == "Normal" else (0, 0, 255)
        cv2.putText(frame, state, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Increment normal or sleepy count based on the state
        if state == "Normal":
            normal_count += 1
        elif state == "Sleepy":
            sleepy_count += 0.1

        if alert_flag:
            # Check if the number of sleepy frames exceeds the number of normal frames
            if sleepy_count > 1.5 *normal_count:
                alert_sound.play()
        
        # Check if 3 seconds have elapsed
        current_time = time.time()
        if current_time - start_time >= 3:
            # Reset counts and start time for the next 3 seconds interval
            normal_count = 0
            sleepy_count = 0
            start_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/video_feed_1')
def video_feed_1():
    return Response(generate_frames_1(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)