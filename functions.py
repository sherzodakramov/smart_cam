import os.path
from datetime import datetime as dt
from uuid import uuid4
import cv2
import numpy as np

# from deepface import DeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion

from database import Database
from deepface.DeepFace import build_model

db = Database()
model_name = 'Facenet512'
models = {"emotion": build_model("Emotion")}

# "age": build_model("Age")
# "emotion": build_model("Emotion"),
# "gender": build_model("Gender"),
# "race": build_model("Race")


def prepare_emotion(current_img, target_size=(224, 224), grayscale=True):
    if current_img.shape[0] > 0 and current_img.shape[1] > 0:
        if grayscale:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        # resize and padding
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            factor_0 = target_size[0] / current_img.shape[0]
            factor_1 = target_size[1] / current_img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(current_img.shape[1] * factor), int(current_img.shape[0] * factor))
            current_img = cv2.resize(current_img, dsize)

            diff_0 = target_size[0] - current_img.shape[0]
            diff_1 = target_size[1] - current_img.shape[1]
            if not grayscale:
                # Put the base image in the middle of the padded image
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                        (0, 0),
                    ),
                    "constant",
                )
            else:
                current_img = np.pad(
                    current_img,
                    ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)),
                    "constant",
                )
        # double check: if target image is not still the same size with target.
        if current_img.shape[0:2] != target_size:
            current_img = cv2.resize(current_img, target_size)
        # normalizing the image pixels
        # img_pixels = current_img.img_to_array(current_img)  # what this line doing? must?
        # img_pixels = np.expand_dims(current_img, axis=0)
        # img_pixels /= 255  # normalize input in [0, 1]
        return current_img


def main_function(face_loc, name, encods, dis, frame, sfr, red_db, enter: int):
    # Unpack face_loc coordinates
    x1, y1, w, h = face_loc
    x2, y2 = x1 + w, y1 + h
    # age = None
    if name == 'Unknown' and int(dis) > 27:
        client_name = f"new-{str(uuid4())}"
        condition = sfr.add_unknown_face(frame[y1 - 10:y2 + 10, x1 - 10:x2 + 10], client_name, encods)
        if condition[0]:
            image_path = f"clients/{client_name}.jpg"
            current_time = dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            # Create a new row for the DataFrame
            new_row = {
                'name': client_name, 'array_bytes': condition[1], 'is_client': 'True',
                'created_time': current_time, 'last_time': current_time,
                'last_enter_time': current_time if enter == 1 else '',
                'last_leave_time': current_time if enter != 1 else '',
                'enter_count': 1 if enter == 1 else 0,
                'leave_count': 1 if enter != 1 else 0,
                'stay_time': 0,
                'image': image_path,
                'last_image': '',
            }
            # cv2.imwrite(image_path, frame[y1 - 10:y2 + 10, x1 - 10:x2 + 10], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # Append the new row to the DataFrame
            # red_db.add_person(person=f"client:{client_name}", **new_row)
            print("Successfully saved")
    else:
        condition = sfr.add_unknown_face(frame[y1 - 10:y2 + 10, x1 - 10:x2 + 10], name, encods)
        if not condition[0]:
            return False
        # Check if the name exists in the DataFrame
        cond = red_db.get_field(name=f"client:{name}", field='last_time')
        # result = DeepFace.analyze(frame, actions=('age',), enforce_detection=False, silent=True)
        # gender = result[0]['dominant_gender']
        # age = result[0]['age']
        if cond:
            current_time = dt.now()
            try:
                last_time = dt.strptime(cond, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                last_time = dt.strptime(cond, '%Y-%m-%dT%H:%M:%S.%f')
            time1 = current_time.time()
            time2 = last_time.time()

            # Calculate the time difference in minutes
            time_diff_minutes = (time1.hour * 60 + time1.minute) - (time2.hour * 60 + time2.minute)
            # Check if the time difference is greater than 2 minutes
            if abs(time_diff_minutes) > 2:
                # Update DataFrame entries for an existing face
                image_path = f"last_images/{name}.jpg"
                cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                row = {
                    'last_time': f"{current_time}",
                    'last_image': image_path,
                }

                if enter == 1:
                    row['last_enter_time'] = f"{current_time}"
                    row['enter_count'] = 1
                else:
                    row['last_leave_time'] = f"{current_time}"
                    row['leave_count'] = 1
                    row['stay_time'] = int(time_diff_minutes)
                # update person
                red_db.update_person(person=f"client:{name}", **row)
                print("Updated!!!")
        else:
            is_client = True
            image_path = f"clients/{name}.jpg"
            if not os.path.exists(image_path):
                image_path = f"employees/{name}.jpg"
                if not os.path.exists(image_path):
                    return False
                is_client = False
            current_time = f"{dt.now()}"

            # Create a new row for the DataFrame
            new_row = {
                'name': name, 'array_bytes': condition[1], 'is_client': f'{is_client}',
                'created_time': current_time, 'last_time': current_time,
                'last_enter_time': current_time if enter == 1 else '',
                'last_leave_time': current_time if enter != 1 else '',
                'enter_count': 1 if enter == 1 else 0,
                'leave_count': 1 if enter != 1 else 0,
                'stay_time': 0,
                'image': image_path,
                'last_image': '',
            }

            # Append the new row to the DataFrame
            red_db.add_person(person=f"client:{name}", **new_row)
            if is_client:
                print(f"{name[:6]} - Client successfully saved!")
            else:
                print(f"{name[:6]} Employee saved!")
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
    # if age:
    #     cv2.putText(frame, f"{name[:5]}-{accuracy:.2f}-{age}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200),
    #                 2)
    # else:
    # Draw a rectangle around the detected face
    # age, gender, emotion
    # obj = {}
    # img_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.resize(img_gray, (48, 48))
    # img_gray = np.expand_dims(img_gray, axis=0)
    # #
    # # # emotion
    # emotion_predictions = models["emotion"].predict(img_gray, verbose=0)[0, :]
    #
    # sum_of_predictions = emotion_predictions.sum()
    # obj["emotion"] = {}
    #
    # for i, emotion_label in enumerate(Emotion.labels):
    #     emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
    #     obj["emotion"][emotion_label] = emotion_prediction
    #
    # obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]
    # age
    # img = prepare_emotion(img_gray)
    # age_predictions = models["age"].predict(img, verbose=0)[0, :]
    # apparent_age = Age.findApparentAge(age_predictions)
    # # int cast is for exception - object of type 'float32' is not JSON serializable
    # obj["age"] = int(apparent_age)
    # # gender
    # gender_predictions = models["gender"].predict(frame[y1:y2, x1:x2], verbose=0)[0, :]
    # obj["gender"] = {}
    # for i, gender_label in enumerate(Gender.labels):
    #     gender_prediction = 100 * gender_predictions[i]
    #     obj["gender"][gender_label] = gender_prediction
    #
    # obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]
    cv2.putText(frame, f"{name[:6]}-{dis:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
