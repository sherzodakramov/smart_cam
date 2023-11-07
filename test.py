from deepface.commons import functions as fn
import logging
import time

import cv2
import face_recognition
import numpy as np

from deepface import DeepFace as dp
from deepface.DeepFace import build_model
from deepface.detectors import FaceDetector
from redis_db import Memory
from simple_facerec import SimpleFacerec
from database import Database
from deepface.commons.distance import findThreshold, findCosineDistance

logging.basicConfig(filename="info.log", level=logging.INFO)

db = Database()
red_db = Memory()
face_recognizer = SimpleFacerec()
# Capture video from your camera
camera = cv2.VideoCapture(1)


# face_distances = []
# for enc in encods:
#     face_distances.append(findCosineDistance(enc, face_encoding))
def process_face(face_encoding, encods, names, model):
    name = "Unknown"
    face_distances = face_recognition.face_distance(encods, face_encoding)
    best_match_index = np.argmin(face_distances)
    if face_distances[best_match_index] < findThreshold(f'{model}', 'euclidean'):
        name = names[best_match_index]
    return name, face_distances[best_match_index]


detector_backend = 'mediapipe'
model_name = 'Facenet512'
model = build_model(model_name)
face_detector = FaceDetector.build_model(detector_backend)
target_size = fn.find_target_size(model_name=model_name)

for k in ['Facenet512']:
    # red_db.open_connection().flushall()
    # db.delete_users()
    # print(f"Flushed: {k}")
    face_recognizer.load_encoding_images("employees/", red_db)
    face_recognizer.load_encoding_images("clients/", red_db)
    red_db.get_all_people('name', 'array_bytes')
    encods = red_db.people_encodings
    names = red_db.people_names
    print(f"Loaded: {k}")
    # end_time = datetime.datetime.now() + datetime.timedelta(seconds=120)
    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        objs = [x for x in dp.represent(image, enforce_detection=False,
                                        detector_backend=detector_backend, model=model,
                                        target_size=target_size, face_detector=face_detector)]
        face_encodings = [np.array(x['embedding'], dtype=np.float64) for x in objs]
        face_locations = [x['facial_area'] for x in objs]

        for face_encod, face_loc in zip(face_encodings, face_locations):
            name, distance = process_face(face_encod, encods, names, model=k)
            x1, y1, w, h = face_loc
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
            cv2.putText(frame, f"{name[:5]}-{(1-distance):.2f}", (x1, y1 - 13), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        frame_count += 1
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:
            break

    # Calculate and display FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    # logging.info(f"Average FPS for {k}: {fps:.2f}")
    print(f"Average FPS for {k}: {fps:.2f}")

camera.release()
cv2.destroyAllWindows()
