import os.path
from datetime import datetime as dt
from uuid import uuid4
# from deepface import DeepFace
import cv2
from database import Database

db = Database()


def main_function(face_loc, name, dis, frame, sfr, enter: int):
    # Unpack face_loc coordinates
    y1, x2, y2, x1 = face_loc
    # Draw a rectangle around the detected face
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
    # age = None
    if name == 'Unknown':
        client_name = str(uuid4())
        if sfr.add_known_face(frame[y1 - 10:y2 + 10, x1 - 10:x2 + 10], client_name):
            image_path = f"clients/{client_name}.jpg"
            current_time = dt.now()

            # Create a new row for the DataFrame
            if enter == 1:
                new_row = {'name': client_name, 'is_client': True, 'created_time': current_time,
                           'last_time': current_time,
                           'last_enter_time': current_time, 'last_leave_time': current_time, 'enter_count': 1,
                           'leave_count': 0, 'stay_time': 0, 'image': image_path, 'last_image': ''}
            else:
                new_row = {'name': client_name, 'is_client': True, 'created_time': current_time,
                           'last_time': current_time,
                           'last_enter_time': current_time, 'last_leave_time': current_time, 'enter_count': 0,
                           'leave_count': 1, 'stay_time': 0, 'image': image_path, 'last_image': ''}
            # Save the cropped face as an image
            cropped_face = frame[y1 - 10:y2 + 10, x1 - 10:x2 + 10]
            cv2.imwrite(image_path, cropped_face)

            # Append the new row to the DataFrame
            db.add_person(**new_row)
            print("Successfully saved")
    else:
        # Check if the name exists in the DataFrame
        if db.select_person(name=name):
            # result = DeepFace.analyze(frame, actions=('age',), enforce_detection=False, silent=True)
            # gender = result[0]['dominant_gender']
            # age = result[0]['age']
            current_time = dt.now()
            last_time = db.select_param(param='last_time', name=name)[0]
            time_diff_minutes = (current_time - last_time).total_seconds() / 60
            if time_diff_minutes > 2:
                # Update DataFrame entries for an existing face
                image_path = f"last_images/{name}.jpg"
                cv2.imwrite(image_path, frame)
                if enter == 1:
                    row = {'last_time': current_time, 'last_enter_time': current_time, 'enter_count': 1,
                           'last_image': image_path}
                else:
                    row = {'last_time': current_time, 'last_leave_time': current_time, 'leave_count': 1,
                           'last_image': image_path, 'stay_time': int(time_diff_minutes)}
                db.update_person(name=name, **row)
                print("Updated!!!")
        else:
            is_client = True
            image_path = f"clients/{name}.jpg"
            if not os.path.exists(image_path):
                image_path = f"employees/{name}.jpg"
                is_client = False
            current_time = dt.now()

            # Create a new row for the DataFrame
            if enter == 1:
                new_row = {'name': name, 'is_client': is_client, 'created_time': current_time,
                           'last_time': current_time,
                           'last_enter_time': current_time, 'last_leave_time': current_time, 'enter_count': 1,
                           'leave_count': 0, 'stay_time': 0, 'image': image_path, 'last_image': ''}
            else:
                new_row = {'name': name, 'is_client': True, 'created_time': current_time,
                           'last_time': current_time,
                           'last_enter_time': current_time, 'last_leave_time': current_time, 'enter_count': 0,
                           'leave_count': 1, 'stay_time': 0, 'image': image_path, 'last_image': ''}

            # Append the new row to the DataFrame
            db.add_person(**new_row)
            if is_client:
                print("Client successfully saved!")
            else:
                print("Employee saved!")
    accuracy = 1-dis
    # if age:
    #     cv2.putText(frame, f"{name[:5]}-{accuracy:.2f}-{age}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200),
    #                 2)
    # else:
    cv2.putText(frame, f"{name[:6]}-{accuracy:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200),
                2)
