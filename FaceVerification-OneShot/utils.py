import cv2
import numpy as np
import tensorflow as tf
from keras_facenet import FaceNet
import pygame
import pickle
import os
import time

# Load FaceNet model
model = FaceNet().model

# Function to convert image to embedding
def img_to_encoding(image, model):
    image = cv2.resize(image, (160, 160))
    img = np.around(np.array(image) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

# Function to verify identity
def verify(image, targets, database, model):
    encoding = img_to_encoding(image, model)
    min_dist = float('inf')
    min_identity = None
    for identity in targets:
        dist = np.linalg.norm(encoding - database[identity], ord=2)
        if dist < min_dist:
            min_dist = dist
            min_identity = identity
    if min_dist < 0.75:
        print("It's " + str(min_identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not one of the targets, please go away")
        door_open = False
    return min_dist, door_open

# Load database
with open('data.pickle', 'rb') as handle:
    database = pickle.load(handle)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

targets = []
with open('data.pickle', 'rb') as handle:
    database = pickle.load(handle)

for person, vector in database.items():
    targets.append(person)

# Variables to track access statistics
access_successful_count = 0
access_denied_count = 0

# Time interval for counting access statistics (10 seconds)
time_interval = 20
start_time = time.time()

# Initialize dist variable
dist = 0

# Initialize pygame mixer
pygame.mixer.init()

# Load access successful sound
success_sound = pygame.mixer.Sound(r"tingting.mp3")

# Load access denied sound
denied_sound = pygame.mixer.Sound(r"sound (mp3cut.net).mp3")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop face region
        face_region = frame[y:y + h, x:x + w]

        # Verify identity
        dist, door_open = verify(face_region, targets, database, model)

        # Increment access counts based on verification result
        if door_open:
            access_successful_count += 1
            cv2.putText(frame, "Access Successful", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        else:
            access_denied_count += 1
            cv2.putText(frame, "Access Denied", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            

    # Display "Distance" outside the loop
    cv2.putText(frame, "Distance: {:.2f}".format(dist), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)

    # Check if the time interval has elapsed
    current_time = time.time()
    if current_time - start_time >= time_interval:
        # Determine which folder to save images based on access statistics
        if access_successful_count > access_denied_count:
            save_folder = "access"
            success_sound.play()
        else:
            save_folder = "denied"
            denied_sound.play()
        
        # Create the folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)

        # Capture and save the image
        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = os.path.join(save_folder, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

        # Wait for the sound to finish playing before stopping the program
        sound_length = max(success_sound.get_length(), denied_sound.get_length())
        pygame.time.delay(int(sound_length * 1000))  # Delay time is in milliseconds
        break

    # Check if 5 seconds have elapsed and print access statistics
    if int(current_time - start_time) % 5 == 0:
        print(f"Access Successful: {access_successful_count}, Access Denied: {access_denied_count}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
