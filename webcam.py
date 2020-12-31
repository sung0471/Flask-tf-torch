import requests
import cv2
import time
from utils.utils import FileControl

path = FileControl()

ip = 'localhost'
url = f"http://{ip}:5000/tracking"
use_webCam = True
if use_webCam:
    file_name = 0
    location = 1
else:
    file_name = path.get_video_path('box.mp4')
    location = 0
video_capture = cv2.VideoCapture(file_name)

payload = {}
headers = {}
idx = 0
start_point = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    if idx >= start_point:
        _, img_encoded = cv2.imencode('.jpg', frame)

        files = [
            ('file', (f'frame{idx}.jpg', img_encoded.tostring(), 'image/jpeg'))
        ]
        # URL 번호 frame에서 time으로 교체 해야함!
        url_full = url + f'?location={location}&frame={idx}'

        check = True
        while check:
            try:
                start = time.time()
                print(url_full)
                response = requests.request("GET", url_full, headers=headers, data=payload, files=files)

                print(response.text)
                print('request time : ', time.time() - start)
                check = False
                idx += 1
            except Exception as e:
                print(e)
                cv2.waitKey(1)
