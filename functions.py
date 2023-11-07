import os.path
from datetime import datetime as dt
from uuid import uuid4
import cv2

from database import Database

db = Database()


def main_function(face_loc, name, dis, encod, frame, red_db, enter: int):
    # Unpack face_loc coordinates
    y1, x2, y2, x1 = face_loc
    # age = None

    if name == 'Unknown':
        client_name = str(uuid4())
        image_path = f"clients/{client_name}.jpg"
        current_time = dt.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        # Create a new row for the DataFrame
        new_row = {
            'name': client_name, 'array_bytes': encod, 'is_client': 'True',
            'created_time': current_time, 'last_time': current_time,
            'last_enter_time': current_time if enter == 1 else '',
            'last_leave_time': current_time if enter != 1 else '',
            'enter_count': 1 if enter == 1 else 0,
            'leave_count': 1 if enter != 1 else 0,
            'stay_time': 0,
            'image': image_path,
            'last_image': '',
        }
        cv2.imwrite(image_path, frame[y1 - 13:y2 + 13, x1 - 13:x2 + 13], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # Append the new row to the DataFrame
        red_db.add_person(person=f"client:{client_name}", **new_row)
        print("Successfully saved")
    else:
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
            if abs(time_diff_minutes) > 5:
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
                'name': name, 'array_bytes': encod, 'is_client': f'{is_client}',
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
    accuracy = 1 - dis
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
    # if age:
    #     cv2.putText(frame, f"{name[:5]}-{accuracy:.2f}-{age}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200),
    #                 2)
    # else:
    # Draw a rectangle around the detected face
    cv2.putText(frame, f"{name}-{accuracy:.2f}", (x1, y1 - 13), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
