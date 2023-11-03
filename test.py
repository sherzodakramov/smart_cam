import datetime
import time

import cv2
import face_recognition
import numpy as np

from deepface import DeepFace as dp
from redis_db import Memory

encods = Memory().get_all_people('name', 'array_bytes')
# Capture video from your camera
camera = cv2.VideoCapture(1)


def process_face(face_encoding):
    name = "Unknown"
    face_distances = face_recognition.face_distance(encods, face_encoding)
    best_match_index = np.argmin(face_distances)
    return name, face_distances[best_match_index]


for k in ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'SFace']:
    end_time = datetime.datetime.now() + datetime.timedelta(seconds=60)
    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_count = 0
    while datetime.datetime.now() < end_time:
        ret, frame = camera.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img_objs = functions.extract_faces(
        #     img=image,
        #     target_size=(224, 224),
        #     detector_backend=k,
        #     grayscale=False,
        #     enforce_detection=False,
        #     align=True,
        # )
        objs = [x for x in dp.represent(image, enforce_detection=False, detector_backend='mediapipe', model_name=k)]
        face_encodings = [np.array(x['embedding'], dtype=np.float64) for x in objs]
        face_locations = [x['facial_area'] for x in objs]

        for face_encod, face_loc in face_encodings, face_locations:
            name, distance = process_face(face_encod)
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
    print(f"Average FPS for {k}: {fps:.2f}")

camera.release()
cv2.destroyAllWindows()
