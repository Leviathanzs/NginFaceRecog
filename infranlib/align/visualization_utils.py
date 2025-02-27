from PIL import ImageDraw


def show_results(img, bounding_boxes, facial_landmarks = []):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline = 'white')

    inx = 0
    left_eye = None
    right_eye = None
    for p in facial_landmarks:
        left_eye = [p[0], p[5]]
        right_eye = [p[1], p[6]]
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], fill ="yellow", outline = 'red')
    
    left_eye_x, left_eye_y = left_eye[0], left_eye[1]
    right_eye_x, right_eye_y = right_eye[0], right_eye[1]
    # print(left_eye_x, left_eye_y, right_eye_x, right_eye_y)

    return img_copy