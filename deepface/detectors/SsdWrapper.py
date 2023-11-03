import os
import gdown
import cv2
import numpy as np
import pandas as pd
from deepface.detectors import OpenCvWrapper
from deepface.commons import functions

# pylint: disable=line-too-long


def build_model():

    home = functions.get_deepface_home()

    # model structure
    if not os.path.isfile(home + "/.deepface/weights/deploy.prototxt"):

        print("deploy.prototxt will be downloaded...")

        url = "https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt"

        output = home + "/.deepface/weights/deploy.prototxt"

        gdown.download(url, output, quiet=False)

    # pre-trained weights
    if not os.path.isfile(home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"):

        print("res10_300x300_ssd_iter_140000.caffemodel will be downloaded...")

        url = ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830"
               "/res10_300x300_ssd_iter_140000.caffemodel")

        output = home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel"

        gdown.download(url, output, quiet=False)

    face_detector = cv2.dnn.readNetFromCaffe(
        home + "/.deepface/weights/deploy.prototxt",
        home + "/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel",
    )

    eye_detector = OpenCvWrapper.build_cascade("haarcascade_eye")

    detector = {"face_detector": face_detector, "eye_detector": eye_detector}

    return detector


def detect_face(detector, img, align=True):
    resp = []

    ssd_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

    target_size = (300, 300)

    base_img = img.copy()  # We will restore base_img to img later
    original_size = img.shape

    img = cv2.resize(img, target_size)
    aspect_ratio_x = original_size[1] / target_size[1]
    aspect_ratio_y = original_size[0] / target_size[0]
    img = img.astype(np.float32) / 255.0  # Convert to float32 and scale pixel values to [0, 1]

    imageBlob = cv2.dnn.blobFromImage(image=img)
    face_detector = detector["face_detector"]
    face_detector.setInput(imageBlob)
    detections = face_detector.forward()

    detections_df = pd.DataFrame(detections[0][0], columns=ssd_labels)

    # Filter only face detections with confidence >= 0.90
    detections_df = detections_df[(detections_df["is_face"] == 1) & (detections_df["confidence"] >= 0.90)]

    if not detections_df.empty:
        for _, instance in detections_df.iterrows():
            left = int(instance["left"] * aspect_ratio_x)
            right = int(instance["right"] * aspect_ratio_x)
            top = int(instance["top"] * aspect_ratio_y)
            bottom = int(instance["bottom"] * aspect_ratio_y)

            detected_face = base_img[top:bottom, left:right]
            img_region = [left, top, right - left, bottom - top]
            confidence = instance["confidence"]

            if align:
                detected_face = OpenCvWrapper.align_face(detector["eye_detector"], detected_face)

            resp.append((detected_face, img_region, confidence))

    return resp
