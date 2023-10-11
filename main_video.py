import cv2
from imutils.video import FPS
from threading import Thread
from simple_facerec import SimpleFacerec
from database import Database
from functions import main_function


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
    db = Database()
    db.create_table_client()

    # Specify the camera numbers you want to use
    camera_numbers = [1]  # Example: Use cameras 0 and 1

    # Create a separate thread for each camera
    threads = []
    for camera_number in camera_numbers:
        thread = Thread(target=camera_process, args=(camera_number,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
