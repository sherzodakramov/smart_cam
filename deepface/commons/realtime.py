import os

import cv2
# import time
import numpy as np
import pandas as pd

from deepface import DeepFace
from deepface.commons import functions

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# pylint: disable=too-many-nested-blocks


def analysis(db_path, model_name="VGG-Face", detector_backend="opencv", distance_metric="cosine",
             enable_face_analysis=True, source=1):
    DeepFace.build_model(model_name=model_name)
    if enable_face_analysis:
        DeepFace.build_model(model_name="Age")
        DeepFace.build_model(model_name="Gender")
        DeepFace.build_model(model_name="Emotion")

    cap = cv2.VideoCapture(source)
    text_color = (255, 255, 255)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        faces = DeepFace.extract_faces(img_path=img, detector_backend=detector_backend, enforce_detection=False)
        for face in faces:
            x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]

            # if w > 60:
            cv2.rectangle(img, (x, y), (x + w, y + h), (67, 67, 67), 1)
            custom_face = img[y:y + h, x:x + w]

            if enable_face_analysis:
                demographies = DeepFace.analyze(img_path=custom_face, detector_backend=detector_backend, enforce_detection=False, silent=True)

                if demographies:
                    demography = demographies[0]
                    if "emotion" in demography and "age" in demography and "dominant_gender" in demography:
                        emotion = demography["emotion"]
                        emotion_df = pd.DataFrame(emotion.items(), columns=["emotion", "score"])
                        emotion_df = emotion_df.sort_values(by=["score"], ascending=False).reset_index(drop=True)

                        analysis_report = f"{int(demography['age'])} {'M' if demography['dominant_gender'] == 'Man' else 'W'}"

                        overlay = img.copy()
                        opacity = 0.4
                        mood_area = (x + w, y) if x + w + 112 < img.shape[1] else (x - 112, y)

                        cv2.rectangle(overlay, mood_area, (mood_area[0] + 112, mood_area[1] + h), (64, 64, 64), cv2.FILLED)
                        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

                        for index, (current_emotion, emotion_score) in emotion_df.iterrows():
                            bar_x = int(35 * emotion_score / 100)
                            text_location_x = mood_area[0] + w if mood_area[0] == x + w else mood_area[0]
                            text_location_y = y + 20 + (index + 1) * 20 if y + 20 + (index + 1) * 20 < y + h else y + h
                            cv2.putText(img, f"{current_emotion} ", (text_location_x, text_location_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                            cv2.rectangle(img, (text_location_x + 70, text_location_y - 7), (text_location_x + 70 + bar_x, text_location_y - 2), (255, 255, 255), cv2.FILLED)

                dfs = DeepFace.find(img_path=custom_face, db_path=db_path, model_name=model_name,
                                    detector_backend=detector_backend, distance_metric=distance_metric,
                                    enforce_detection=False, silent=True)

                if dfs:
                    df = dfs[0]
                    if not df.empty:
                        candidate = df.iloc[0]
                        label = candidate["identity"].split("/")[-1]

                        display_img = cv2.imread(label)
                        source_objs = DeepFace.extract_faces(img_path=label, target_size=(112, 112),
                                                             detector_backend=detector_backend, enforce_detection=False, align=True)

                        if source_objs:
                            source_obj = source_objs[0]
                            display_img = source_obj["face"] * 255
                            display_img = display_img[:, :, ::-1]

                        if y - 112 > 0 and mood_area[0] == x + w:
                            top_area = (x + int(w / 2) - int(w / 10), y - 37, x + int(w / 2) + int(w / 10), y - 37 + 112)
                            mood_text_location = (x + int(w / 3.5), y - 55)
                        elif y + h + 112 < img.shape[0] and mood_area[0] == x - 112:
                            top_area = (x + int(w / 2) - int(w / 10), y + h + 37, x + int(w / 2) + int(w / 10), y + h + 37 + 112)
                            mood_text_location = (x + int(w / 3.5), y + h + 85)
                        elif y - 112 > 0 and mood_area[0] == x - 112:
                            top_area = (x - 112, y, x, y + 112)
                            mood_text_location = (x - 80, y + 10)
                        else:
                            top_area = (x + w, y + h, x + w + 112, y + h + 112)
                            mood_text_location = (x + w, y + h + 50)

                        triangle_coordinates = np.array([
                            (x + int(w / 2), y), (top_area[0] + int((top_area[2] - top_area[0]) / 2), top_area[1]),
                            (x + int(w / 2), top_area[1])
                        ])

                        cv2.drawContours(img, [triangle_coordinates], 0, (46, 200, 255), -1)
                        cv2.rectangle(img, (top_area[0] + int((top_area[2] - top_area[0]) / 5), top_area[1]), (top_area[2] - int((top_area[2] - top_area[0]) / 5), top_area[3]), (46, 200, 255), cv2.FILLED)
                        cv2.putText(img, analysis_report, mood_text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
                        line_x = x + int(w / 2)
                        line_y1, line_y2 = y, top_area[1]
                        cv2.line(img, (line_x, line_y1), (line_x + int(w / 4), line_y1 - 56), (67, 67, 67), 1)
                        cv2.line(img, (line_x + int(w / 4), line_y1 - 56), (line_x + w, line_y1 - 56), (67, 67, 67), 1)

                img[top_area[1]:top_area[3], top_area[0]:top_area[2]] = display_img
                overlay = img.copy()
                opacity = 0.4
                mood_bar_area = (top_area[0] + (top_area[2] - top_area[0]) / 5, top_area[1], top_area[0] + (top_area[2] - top_area[0]) / 5 + 35, top_area[1] + 37)
                cv2.rectangle(overlay, mood_bar_area, (top_area[2], top_area[3]), (46, 200, 255), cv2.FILLED)
                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                cv2.putText(img, label, (top_area[0], top_area[1] + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                line_x = x + int(w / 2)
                line_y1, line_y2 = top_area[1], top_area[1] - 56
                cv2.line(img, (line_x, line_y1), (line_x - int(w / 4), line_y2), (67, 67, 67), 1)
                cv2.line(img, (line_x - int(w / 4), line_y2), (line_x, line_y2), (67, 67, 67), 1)
            dfs = DeepFace.find(img_path=custom_face, db_path=db_path, model_name=model_name,
                                detector_backend=detector_backend, distance_metric=distance_metric,
                                enforce_detection=False, silent=True)
            print(dfs)
        cv2.rectangle(img, (10, 10), (90, 50), (67, 67, 67), -10)
        cv2.imshow("img", img)

        if cv2.waitKey(1) == 27:  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
