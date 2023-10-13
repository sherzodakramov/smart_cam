import os.path
from datetime import datetime as dt
from uuid import uuid4
import cv2
import psycopg2
from deepface import DeepFace

from database import Database

db = Database()


def main_function(face_loc, name, dis, frame, sfr, enter: int):
    # Unpack face_loc coordinates
    y1, x2, y2, x1 = face_loc
    age = None

    if name == 'Unknown':
        client_name = str(uuid4())
        condition = sfr.add_known_face(frame[y1 - 13:y2 + 13, x1 - 13:x2 + 13], client_name)
        if condition[0]:
            image_path = f"clients/{client_name}.jpg"
            current_time = dt.now()
            encoded = psycopg2.Binary(condition[1].tobytes())

            # Create a new row for the DataFrame
            new_row = {
                'name': client_name, 'array_bytes': encoded, 'is_client': True,
                'created_time': current_time, 'last_time': current_time,
                'last_enter_time': current_time if enter == 1 else None,
                'last_leave_time': current_time if enter != 1 else None,
                'enter_count': 1 if enter == 1 else 0,
                'leave_count': 1 if enter != 1 else 0,
                'stay_time': 0,
                'image': image_path,
                'last_image': '',
            }
            cv2.imwrite(image_path, frame[y1 - 13:y2 + 13, x1 - 13:x2 + 13], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # Append the new row to the DataFrame
            db.add_person(**new_row)
            print("Successfully saved")
    else:
        condition = sfr.add_known_face(frame[y1 - 13:y2 + 13, x1 - 13:x2 + 13], name)
        # Check if the name exists in the DataFrame
        cond = db.select_param(param='last_time', name=name)
        result = DeepFace.analyze(frame, actions=('age',), enforce_detection=False, silent=True)
        # gender = result[0]['dominant_gender']
        age = result[0]['age']
        if cond:
            current_time = dt.now()
            last_time = cond[0]
            time_diff_minutes = (current_time - last_time).total_seconds() / 60
            if time_diff_minutes > 2:
                # Update DataFrame entries for an existing face
                image_path = f"last_images/{name}.jpg"
                cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                row = {
                    'last_time': current_time,
                    'last_image': image_path,
                }

                if enter == 1:
                    row['last_enter_time'] = current_time
                    row['enter_count'] = 1
                else:
                    row['last_leave_time'] = current_time
                    row['leave_count'] = 1
                    row['stay_time'] = int(time_diff_minutes)

                db.update_person(name=name, **row)
                print("Updated!!!")
        else:
            is_client = True
            image_path = f"clients/{name}.jpg"
            if not os.path.exists(image_path):
                image_path = f"employees/{name}.jpg"
                if not os.path.exists(image_path):
                    return False
                is_client = False
            current_time = dt.now()

            # Create a new row for the DataFrame
            new_row = {
                'name': name, 'array_bytes': condition[1], 'is_client': True,
                'created_time': current_time, 'last_time': current_time,
                'last_enter_time': current_time if enter == 1 else None,
                'last_leave_time': current_time if enter != 1 else None,
                'enter_count': 1 if enter == 1 else 0,
                'leave_count': 1 if enter != 1 else 0,
                'stay_time': 0,
                'image': image_path,
                'last_image': '',
            }

            # Append the new row to the DataFrame
            db.add_person(**new_row)
            if is_client:
                print(f"{name[:6]} - Client successfully saved!")
            else:
                print(f"{name[:6]} Employee saved!")
    accuracy = 1 - dis
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
    if age:
        cv2.putText(frame, f"{name[:5]}-{accuracy:.2f}-{age}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200),
                    2)
    else:
    # Draw a rectangle around the detected face
        cv2.putText(frame, f"{name[:4]}-{accuracy:.2f}", (x1, y1 - 13), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
