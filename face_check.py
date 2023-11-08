import math
import face_recognition


def correct_faces(frame, locations):
    face_landmarks_list = face_recognition.face_landmarks(frame, face_locations=locations)
    correct_face_locs = []
    # Assuming only one face is detected
    for i, landmarks in enumerate(face_landmarks_list):
        y1, x2, y2, x1 = locations[i]
        face = frame[y1:y2, x1:x2]
        if face.shape[0] < 70 or face.shape[0] < 70:
            continue
        # Calculate key facial landmark positions
        # nose = landmarks["nose_bridge"][3]  # Adjust index as needed
        # chin = landmarks["chin"][10]  # Adjust index as needed
        left_eye = landmarks["left_eye"][1]  # Adjust index as needed
        x1, y1 = left_eye
        right_eye = landmarks["right_eye"][2]  # Adjust index as needed
        x2, y2 = right_eye
        nose = landmarks["nose_tip"][2]  # Adjust index as needed
        x3, y3 = nose
        mouth = landmarks["top_lip"][9]  # Adjust index as needed
        x4, y4 = mouth
        lr = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        r = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        l = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
        # nose_to_mouth = math.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
        alfa = (l**2 + r**2 - lr**2)/(2*l*r)

        if not (alfa >= 0.4 or alfa <= -0.35):
            correct_face_locs.append(locations[i])
    return correct_face_locs
