import glob
import logging
import os
from datetime import datetime as dt, datetime

import cv2
import concurrent.futures
import face_recognition
import numpy as np
import psycopg2

from database import Database

logging.basicConfig(filename="info.log", level=logging.INFO)
n_jitter = 100

db = Database()


class SimpleFacerec:
    def __init__(self):
        self.frame_resizing = 1

    def add_unknown_face(self, image, name):
        try:
            if image is None:
                # logging.error("Image is None.")
                return [False]

            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_img, num_jitters=n_jitter, model='large')

            if len(face_encodings) > 0:
                img_encoding = face_encodings[0]
                return [True, img_encoding]
            else:
                # logging.warning(f"No face found for {name}.")
                return [False]
        except Exception as e:
            logging.exception(f"{datetime.now()}: Error while adding known face for {name}: {str(e)}")
            return [False]

    def load_encoding_images(self, images_path, red_db):
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
                if len(encod) == 0:
                    return None
                is_client = folder != 'employees'
                new_row = {'name': filename, 'array_bytes': psycopg2.Binary(encod[0].tobytes()),
                           'is_client': f"{is_client}", 'created_time': current_time, 'last_time': current_time,
                           'last_enter_time': current_time, 'last_leave_time': current_time, 'enter_count': 1,
                           'leave_count': 0, 'stay_time': 0, 'image': img_path, 'last_image': ''}
                db.add_person(**new_row)
                in_memory = red_db.get_field(name=f"client:{filename}", field='array_bytes')
                if not in_memory:
                    time = current_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
                    new_row['array_bytes'] = encod[0]
                    new_row['created_time'] = time
                    new_row['last_time'] = time
                    new_row['last_enter_time'] = time
                    new_row['last_leave_time'] = time
                    red_db.add_person(person=f"client:{filename}", **new_row)
            else:
                if not face_encodings[0]:
                    encoded = face_recognition.face_encodings(rgb_img, num_jitters=n_jitter, model='large')[0]
                    db.update_person(name=filename, **{'array_bytes': psycopg2.Binary(encoded.tobytes())})
                    red_db.update_person(person=f"client:{filename}", **{'array_bytes': f"{encoded}"})
                else:
                    row = {'name': filename, 'array_bytes': f"{np.frombuffer(face_encodings[0], dtype=np.float64)}"}
                    red_db.update_person(person=f"client:{filename}",
                                         **row)
            return True

        for img_path in images_path:
            img = cv2.imread(img_path)
            try:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except cv2.error:
                continue
            if not process_image(img_path):
                try:
                    os.remove(img_path)
                    logging.warning(f"{datetime.now()}: {img_path} has been successfully removed.")
                except OSError as e:
                    print(f"Error: {e}")

        print("Encoding images loaded")

    def detect_known_faces(self, frame, names, encods, accuracy: float = 0.48):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Use the 'cnn' model for more accurate face detection
        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=n_jitter,
                                                         model='large')

        def process_face(face_encoding):
            name = "Unknown"
            face_distances = face_recognition.face_distance(encods, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] <= (1 - accuracy):
                name = names[best_match_index]

            return name, face_distances[best_match_index]

        if len(face_encodings) > 1:
            # apply with multithreading
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(process_face, face_encodings))
            if results:
                face_names, distances = zip(*results)
            else:
                face_names = []
                distances = []
                # Handle the case when results are empty
        else:
            face_names = []
            distances = []
            for face in face_encodings:
                face_name, face_dis = process_face(face)
                face_names.append(face_name)
                distances.append(face_dis)

        face_locations = (np.array(face_locations) / self.frame_resizing).astype(int)
        return face_locations, face_names, distances
