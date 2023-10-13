import glob
import logging
import os
from datetime import datetime as dt

import cv2
import face_recognition
import numpy as np
import psycopg2

from database import Database

logging.basicConfig(filename="info.log", level=logging.INFO)
n_jitter = 100

db = Database()


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 1

    def add_known_face(self, image, name):
        try:
            if image is None:
                logging.error("Image is None.")
                return [False]

            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_img, num_jitters=n_jitter, model='large')

            if len(face_encodings) > 0:
                img_encoding = face_encodings[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(name)
                return [True, img_encoding]
            else:
                logging.warning(f"No face found for {name}.")
                return [False]
        except Exception as e:
            logging.exception(f"Error while adding known face for {name}: {str(e)}")
            return [False]

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

        def process_image(img_path):
            folder = os.path.dirname(img_path)
            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)
            face_encodings = db.select_param('array_bytes', name=filename)

            if not face_encodings:
                current_time = dt.now()
                encod = face_recognition.face_encodings(rgb_img, num_jitters=n_jitter, model='large')
                if len(encod) > 0:
                    encoded = encod[0]
                else:
                    return None
                self.known_face_encodings.append(encoded)
                self.known_face_names.append(filename)
                is_client = folder != 'employees'
                new_row = {'name': filename, 'array_bytes': psycopg2.Binary(encoded.tobytes()), 'is_client': is_client,
                           'created_time': current_time, 'last_time': current_time, 'last_enter_time': current_time,
                           'last_leave_time': current_time, 'enter_count': 1, 'leave_count': 0, 'stay_time': 0,
                           'image': img_path, 'last_image': ''}
                db.add_person(**new_row)
            else:
                if not face_encodings[0]:
                    encoded = face_recognition.face_encodings(rgb_img, num_jitters=n_jitter, model='large')[0]
                    self.known_face_encodings.append(encoded)
                    self.known_face_names.append(filename)
                    new_row = {'array_bytes': psycopg2.Binary(encoded.tobytes())}
                    db.update_person(name=filename, **new_row)
                else:
                    self.known_face_encodings.append(np.frombuffer(face_encodings[0], dtype=np.float64))
                    self.known_face_names.append(filename)
            return True

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not process_image(img_path):
                try:
                    os.remove(img_path)
                    logging.warning(f"{img_path} has been successfully removed.")
                except OSError as e:
                    print(f"Error: {e}")

        print("Encoding images loaded")

    def detect_known_faces(self, frame, accuracy: float = 0.48):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=n_jitter,
                                                         model='large')
        distances = []
        face_names = []

        for face_encoding in face_encodings:
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] <= (1 - accuracy):
                name = self.known_face_names[best_match_index]

            face_names.append(name)
            distances.append(face_distances[best_match_index])

        face_locations = (np.array(face_locations) / self.frame_resizing).astype(int)
        return face_locations, face_names, distances
