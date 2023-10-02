# import concurrent.futures
import os

import cv2
import pandas as pd
from imutils.video import FPS

from functions import first_cam
from simple_facerec import SimpleFacerec


def main():
    # Encode faces from a folder
    sfr = SimpleFacerec()
    sfr.load_encoding_images("employees/")
    sfr.load_encoding_images("clients/")

    # Load Camera
    cap = cv2.VideoCapture(0)
    if not os.path.exists('data.csv'):
        new_df = pd.DataFrame(
            columns=['name', 'is_client', 'created_time', 'last_time', 'last_enter_time', 'last_leave_time',
                     'enter_count', 'leave_count', 'stay_time', 'image', 'last_image'])
        new_df.to_csv('data.csv', index=False)

    df = pd.read_csv('data.csv')
    df['created_time'] = pd.to_datetime(df['created_time'])
    df['last_time'] = pd.to_datetime(df['last_time'])
    df['last_enter_time'] = pd.to_datetime(df['last_enter_time'])
    df['last_leave_time'] = pd.to_datetime(df['last_leave_time'])

    fps = FPS().start()  # Start the FPS counter before entering the loop

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect Faces
        face_locations, face_names, distances = sfr.detect_known_faces(frame)

        # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for number in range(len(face_locations)):
            df = first_cam(face_locations[number], face_names[number], distances[number], frame, sfr, df)
            # df = executor.submit(first_cam, face_locations[number], face_names[number], distances[number], frame,
            #                      sfr, df).result()

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        fps.update()  # Update FPS counter after processing each frame

    fps.stop()  # Stop the FPS counter when exiting the loop

    df.to_csv('data.csv', index=False)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Elapsed time: {round(fps.elapsed(), 2)} seconds")
    print(f"FPS: {round(fps.fps(), 2)}")


if __name__ == "__main__":
    main()
