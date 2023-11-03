import datetime

import cv2
import numpy as np
import ultralytics
import time

# Load YOLO model
model = ultralytics.YOLO('yolo_mod/best.pt')

# Capture video from your camera
camera = cv2.VideoCapture(1)

end_time = datetime.datetime.now() + datetime.timedelta(seconds=25)
# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

while datetime.datetime.now() < end_time:
    ret, frame = camera.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference with YOLO model
    locs = model.predict(source=image)[0].boxes.xyxy.tolist()

    for face_loc in locs:
        face_loc = np.array(face_loc).astype(int)
        x1, y1, x2, y2 = face_loc
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

    cv2.imshow("Face Recognition", frame)

    frame_count += 1

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()

# Calculate and display FPS
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time
print(f"Average FPS: {fps:.2f}")
