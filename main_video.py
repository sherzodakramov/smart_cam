import threading
from datetime import datetime
from multiprocessing import Process

import cv2
import numpy as np
import pandas as pd
from imutils.video import FPS
from simple_facerec import SimpleFacerec
from database import Database
from functions import main_function
import time
import redis

# Define the time to schedule the database saving (e.g., 3:00 AM)
db = Database()
red = redis.Redis(host='localhost', port=6379, decode_responses=True)

SCHEDULED_TIME = "18:19"


def schedule_database_saving(db, r):
    while True:
        current_time = time.strftime("%H:%M")
        if current_time == SCHEDULED_TIME:
            data_dict = {}
            print("Backup to database is began!!!")
            date_format1 = "%Y-%m-%d %H:%M:%S.%f"
            date_format2 = "%Y-%m-%dT%H:%M:%S.%f"
            # Get all keys (hash names) from Redis
            redis_keys = r.keys('client*')  # Adjust the pattern if necessary

            for key in redis_keys:
                # Fetch data for each Redis hash
                redis_data = r.hgetall(key)
                data_dict[key] = redis_data
                if redis_data:
                    # Check if a record with the same name already exists in the database
                    existing_record = db.select_param('name', redis_data.get('name'))
                    # Convert and call the add_person method
                    name = redis_data['name']
                    array_bytes = np.array(redis_data['array_bytes'].strip('[]').split(), dtype=np.float64)
                    is_client = bool(redis_data['is_client'])
                    try:
                        created_time = datetime.strptime(redis_data['created_time'], date_format1)
                        last_time = datetime.strptime(redis_data['last_time'], date_format1)
                        last_enter_time = datetime.strptime(redis_data['last_enter_time'], date_format1)
                        last_leave_time = datetime.strptime(redis_data['last_leave_time'], date_format1)
                    except:
                        created_time = datetime.strptime(redis_data['created_time'], date_format2)
                        last_time = datetime.strptime(redis_data['last_time'], date_format2)
                        last_enter_time = datetime.strptime(redis_data['last_enter_time'], date_format2)
                        last_leave_time = datetime.strptime(redis_data['last_leave_time'], date_format2)
                    enter_count = int(redis_data['enter_count'])
                    leave_count = int(redis_data['leave_count'])
                    stay_time = int(redis_data['stay_time'])
                    image = redis_data['image']
                    last_image = redis_data['last_image']

                    if existing_record:
                        # Update the existing record
                        continue
                    else:
                        db.add_person(name, array_bytes, is_client, created_time, last_time, last_enter_time,
                                      last_leave_time, enter_count, leave_count, stay_time, image, last_image)

            print("All data saved to the database from Redis.")
            df = pd.DataFrame.from_dict(data_dict, orient='index')

            # Export to Excel
            df.to_excel('clients.xlsx', index=False, engine='openpyxl')
            print("Data converted to excel successfully!!!")
        time.sleep(60)  # Check every minute


def process_frame(face_recognizer, frame, camera_number):
    face_locations, face_names, distances = face_recognizer.detect_known_faces(frame)

    for count in range(len(face_locations)):
        main_function(face_locations[count], face_names[count], distances[count], frame, face_recognizer,
                      enter=camera_number)

    cv2.imshow(f"Camera {camera_number}", frame)
    key = cv2.waitKey(1)

    if key == 27:
        return False  # Exit the processing loop

    return True  # Continue processing frames


def camera_process(camera_number):
    face_recognizer = SimpleFacerec()
    face_recognizer.load_encoding_images("employees/")
    face_recognizer.load_encoding_images("clients/")

    cap = cv2.VideoCapture(camera_number)
    fps = FPS().start()

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Could not read frame from camera {camera_number}.")
            break

        if not process_frame(face_recognizer, frame, camera_number):
            break

        fps.update()

    cv2.destroyAllWindows()
    fps.stop()

    print(f"FPS for Camera {camera_number}: {round(fps.fps(), 2)}")


if __name__ == "__main__":
    db.create_table_client()

    # Specify the camera numbers you want to use
    camera_numbers = [1]  # Example: Use cameras 0 and 1

    # Create a separate process for each camera
    processes = []
    for camera_number in camera_numbers:
        process = Process(target=camera_process, args=(camera_number,))
        processes.append(process)
        process.start()

    # Create a separate thread for the schedule_database_saving function
    save_thread = threading.Thread(target=schedule_database_saving, args=(db, red))

    # Start the thread
    save_thread.start()

    # Wait for all camera processes to finish
    for process in processes:
        process.join()
