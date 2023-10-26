import glob
import logging
import os
from datetime import datetime as dt

import cv2
import concurrent.futures
import face_recognition
import numpy as np
import psycopg2
import knn_algorithm as knn

from database import Database

logging.basicConfig(filename="info.log", level=logging.INFO)
n_jitter = 10

db = Database()


class SimpleFacerec:
    def __init__(self):
        self.frame_resizing = 1

    def add_unknown_face(self, image, name, encoding):
        try:
            if image is None:
                return [False]

            if len(encoding) > 0:
                return [True, encoding]
            else:
                return [False]
        except Exception as e:
            logging.exception(f"{dt.now()}: Error while adding known face for {name}: {str(e)}")
            return [False]

    def detect_known_faces(self, frame, model, distance: float = 0.48):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Use the 'cnn' model for more accurate face detection
        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=n_jitter,
                                                         model='large')

        face_names = []
        distances = []
        for loc, encod in zip(face_locations, face_encodings):
            face_name, face_dis = knn.predict(encod, knn_clf=model, distance_threshold=distance)
            face_names.append(face_name)
            distances.append(face_dis)

        face_locations = (np.array(face_locations) / self.frame_resizing).astype(int)
        return face_locations, face_names, distances, face_encodings
