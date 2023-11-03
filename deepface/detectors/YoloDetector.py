def build_model():
    import ultralytics
    ultralytics.checks()
    model = ultralytics.YOLO('yolo_mod/best.pt')
    return model


def detect_face(yolo_model, img, align=True):
    resp = []

    # img_width = img.shape[1]
    # img_height = img.shape[0]

    # Use the YOLO model to detect faces
    locs = yolo_model.predict(source=img)[0].boxes.xyxy.tolist()

    for face_loc in locs:
        x1, y1, x2, y2 = face_loc
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Create a bounding box object similar to MediaPipe's bounding_box = {"xmin": x1 / img_width, "ymin": y1 /
        # img_height, "xmax": x2 / img_width, "ymax": y2 / img_height}

        # Calculate face size (width and height)
        w = x2 - x1
        h = y2 - y1

        # Calculate confidence score (you may need to adjust this depending on YOLO's output format)
        confidence = 1.0  # You can set it to 1 or use YOLO's confidence score

        # Create a placeholder for landmarks (since YOLO doesn't provide landmarks)
        # landmarks = [0.0, 0.0]  # You can set them to 0 or any other value

        # If you want to align the face, you can do so here (similar to your MediaPipe function)

        detected_face = img[y1: y2, x1: x2]
        img_region = [x1, y1, w, h]

        resp.append((detected_face, img_region, confidence))

    return resp
