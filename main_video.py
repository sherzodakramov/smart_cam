import logging
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import psycopg2
import redis
from imutils.video import FPS

from database import Database
from functions import main_function
from redis_db import Memory
from simple_facerec import SimpleFacerec

logging.basicConfig(filename="info.log", level=logging.INFO)


class Face_App:
    def __init__(self, cameras: list, scheduled_time="11:47"):
        self.cameras = cameras
        self.scheduled_time = scheduled_time
        self.db = Database(host="localhost", database="smart_cam", user="postgres", password="abdu3421")
        self.redis_base = Memory()
        self.face_recognizer = SimpleFacerec()
        self.red = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.stop_flag = False
        self.camera_threads = []

    def stop(self):
        self.stop_flag = True
        for thread in self.camera_threads:
            thread.join()
        print("Successfully stopped!!!")

    def restart(self):
        self.stop()
        self.run_function()

    def schedule_database_saving(self):
        while not self.stop_flag:
            current_time = time.strftime("%H:%M")
            if current_time == self.scheduled_time:
                data_dict = {}
                print("Backup to database is began!!!")
                date_format1 = "%Y-%m-%d %H:%M:%S.%f"
                date_format2 = "%Y-%m-%dT%H:%M:%S.%f"
                # Get all keys (hash names) from Redis
                redis_keys = self.red.keys('client*')  # Adjust the pattern if necessary

                for key in redis_keys:
                    # Fetch data for each Redis hash
                    redis_data = self.red.hgetall(key)
                    data_dict[key] = redis_data
                    if redis_data:
                        # Check if a record with the same name already exists in the database
                        existing_record = self.db.select_param('name', redis_data.get('name'))
                        # Convert and call the add_person method
                        name = redis_data['name']
                        array_bytes = psycopg2.Binary(
                            (np.array(redis_data['array_bytes'].strip('[]').split(), dtype=np.float64).tobytes()))
                        is_client = bool(redis_data['is_client'])

                        def parse_date(date_string, format1, format2):
                            if date_string:
                                try:
                                    return datetime.strptime(date_string, format1)
                                except ValueError:
                                    return datetime.strptime(date_string, format2)
                            else:
                                return datetime.now()

                        created_time = parse_date(redis_data['created_time'], date_format1, date_format2)
                        last_time = parse_date(redis_data['last_time'], date_format1, date_format2)
                        status = True
                        last_enter_time = parse_date(redis_data['last_enter_time'], date_format1, date_format2)
                        last_leave_time = parse_date(redis_data['last_leave_time'], date_format1, date_format2)
                        enter_count = int(redis_data['enter_count'])
                        leave_count = int(redis_data['leave_count'])
                        stay_time = int(redis_data['stay_time'])
                        image = redis_data['image']
                        last_image = redis_data['last_image']
                        dict_person = {'created_time': created_time, 'last_time': last_time,
                                       'last_enter_time': last_enter_time, 'last_leave_time': last_leave_time,
                                       'enter_count': enter_count, 'leave_count': leave_count, 'stay_time': stay_time,
                                       'image': image, 'last_image': last_image}
                        if existing_record:
                            self.db.update_person(name, **dict_person)
                            # Update the existing record
                            continue
                        else:
                            self.db.add_person(name, status, array_bytes, is_client, created_time, last_time,
                                               last_enter_time,
                                               last_leave_time, enter_count, leave_count, stay_time, image, last_image)
                logging.info(f"{datetime.now()}: Database updated!!!")
                print(f"{datetime.now()}: Database updated!!!")
                # df = pd.DataFrame.from_dict(data_dict, orient='index')
                #
                # # Export to Excel
                # df.to_excel('clients.xlsx', index=False, engine='openpyxl')
                # logging.info(f"{datetime.now()}: Data converted to excel successfully!!!")
            time.sleep(60)  # Check every minute

    def process_frame(self, frame, camera_param):
        face_locations, face_names, distances, face_encodings = (
            self.face_recognizer.detect_known_faces(frame, self.redis_base.people_names,
                                                    self.redis_base.people_encodings))
        for count in range(len(face_locations)):
            main_function(face_locations[count], face_names[count], distances[count], face_encodings[count], frame,
                          self.redis_base, enter=camera_param['is_enter'])
        cv2.imshow(f"Camera {camera_param['is_enter']}", frame)
        key = cv2.waitKey(1)

        if key == 27:
            self.stop()
            return False  # Exit the processing loop

        return True  # Continue processing frames

    def camera_process(self, camera_param):
        address = (f"rtsp://{camera_param['login']}:{camera_param['password']}"
                   f"@{camera_param['ip_address']}:554/Streaming/channels/1/")
        cap = cv2.VideoCapture(1)
        fps = FPS().start()

        while not self.stop_flag:
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Could not read frame from camera: {address}.")
                logging.warning(f"Error: Could not read frame from camera: {address}.")
                break

            if not self.process_frame(frame, camera_param):
                break

            fps.update()

        cv2.destroyAllWindows()
        fps.stop()

        print(f"FPS for Camera {address}: {round(fps.fps(), 2)}")

    def run_function(self):
        # initialize database
        self.db.create_table_client()
        # get all people as variable
        self.redis_base.get_all_people('name', 'array_bytes')
        # synchronize redis and database
        self.face_recognizer.load_encoding_images("employees/", self.redis_base)
        # self.face_recognizer.load_encoding_images("clients/", redis_base)

        # Create a thread for each camera
        for camera_param in self.cameras:
            thread = threading.Thread(target=self.camera_process, args=(camera_param,))
            self.camera_threads.append(thread)
            thread.start()

        # Create a separate thread for the schedule_database_saving function
        save_thread = threading.Thread(target=self.schedule_database_saving)

        # Start the thread
        save_thread.start()
        self.camera_threads.append(save_thread)

camera_list = [{'ip_address': '192.168.1.64', 'login': 'admin', 'password': 'softex2020', 'is_enter': True, 'real': 1}]
my_app = Face_App(cameras=camera_list)
my_app.run_function()
