import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=15, refine_landmarks=True, min_tracking_confidence=0.7,
                                  min_detection_confidence=0.7)

while True:
    # Image
    ret, image = cap.read()
    if ret is not True:
        break
    height, width, _ = image.shape
    # image = cv2.imread("employees/Abduaziz.jpg")
    # Face Mesh
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Facial landmarks
    result = face_mesh.process(rgb_image)
    height, width, _ = image.shape
    try:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(image, (x, y), 1, (100, 100, 0), -1)
    except:
        continue
    cv2.imshow("Image", image)
    if cv2.waitKey(1) == 27:
        break
