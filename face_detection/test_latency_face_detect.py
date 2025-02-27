import face_detector_core
import time
from PIL import Image
import base64
import io
import sys

arg = sys.argv

if len(arg) < 2:
    print("How to Use : python test_latency_face_detect.py <name img file>")
else:

    face_detector_core.mtcnn_detector()
    face_detector_core.get_reference()

    startt = time.time() * 1000
    img_str = ''
    try:
        image = Image.open(arg[1])
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_warp, img_str, err_code = face_detector_core.face_alignment(img_str)
        message = 'Success Face Detection'

    except Exception as err:
        print(err)
        
        message = "Error Face Detection Model"
        err_code = 1

    endt = time.time() * 1000
    duration = round(endt - startt)
    print(f"Face Detection time server: {duration} milliseconds")