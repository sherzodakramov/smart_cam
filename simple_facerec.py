import glob
import os
import logging

import cv2
import face_recognition
import numpy as np

logging.basicConfig(filename="info.log", level=logging.INFO)


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 1  # Resize frame for a faster speed

    def add_known_face(self, image, name):
        try:
            if image is not None:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_img, num_jitters=20, model='large')

                if len(face_encodings) > 0:
                    img_encoding = face_encodings[0]
                    self.known_face_encodings.append(img_encoding)
                    self.known_face_names.append(name)
                    return True
                else:
                    logging.warning(f"No face found for {name}.")
                    return False
            else:
                logging.error("Image is None.")
                return False
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return False

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)

            face_encodings = face_recognition.face_encodings(rgb_img, num_jitters=20, model='large')

            if len(face_encodings) > 0:
                img_encoding = face_encodings[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
            else:
                try:
                    os.remove(img_path)
                except (FileNotFoundError, Exception) as e:
                    print(f"An error occurred: {str(e)}")
                # print(f"{filename} is not found!")

        print("Encoding images loaded")

    def detect_known_faces(self, frame, accuracy: float = 0.48):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=20, model='large')

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

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names, distances
