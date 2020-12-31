import os
import numpy as np
import cv2
import io
from base64 import encodebytes
from flask import request
from PIL import Image
from glob import glob

class FileControl:
    """
    media관련 path를 return해주는 함수들을 갖고 있는 class
    """
    def __init__(self):
        this_dir = os.path.abspath(os.path.dirname(__file__))
        base_dir = os.path.abspath(os.path.dirname(this_dir))

        media_dir = os.path.join(base_dir, 'media')
        if not os.path.exists(os.path.join(media_dir, 'image')):
            os.makedirs(os.path.join(media_dir, 'image'))
        self.image_dir = os.path.join(media_dir, 'image')
        self.video_dir = os.path.join(media_dir, 'video')

    def get_image_path(self, location, frame, return_join=True):
        location = str(location)
        location = (4 - len(location)) * '0' + location

        frame = str(frame)
        frame = (8 - len(frame)) * '0' + frame

        image_path = location + frame + '.jpg'

        if return_join:
            return os.path.join(self.image_dir, image_path)
        else:
            return self.image_dir, image_path

    def get_video_path(self, video_name):
        video_path = os.path.join(self.video_dir, video_name)
        return video_path


def get_param_parsing():
    """Get_parsing

    Returns:
        location ([int]): 요청된 location
        time ([str]): 요청된 이미지에 대한 시간 정보
    """
    args = request.args
    param_name = ['location', 'time', 'onlyimage']
    param_default = [0, '0000_00_00_00_00_00_00', False,False]
    param_type = [int, str, bool]
    params = []
    for name, default, data_type in zip(param_name, param_default, param_type):
        params.append(args.get(name, default=default, type=data_type))

    return params

def get_location_parsing():
    """Get_parsing

    Returns:
        location ([int]): 요청된 location
    """
    args = request.args
    return args.get('location', default=0, type=int)

def get_time_list(location):
    """
    Args:
        location ([int]): 요청된 location

    Returns:
        ([str list]): 특정 location에 대한 time_list
    """
    image_paths = glob.glob(f'./media/tracking/{location}/*.jpg')
    return [image_path.replace('.jpg','') for image_path in image_paths]


def get_image():
    """Get_parsing

    Returns:
        file ([np.array]): 요청된 이미지
    """
    image = request.files['file'].read()
    npimg = np.fromstring(image, np.uint8)
    img = cv2.imdecode(np.frombuffer(npimg, dtype=np.uint8), cv2.IMREAD_COLOR)
    #img = Image.fromarray(img, 'RGB')

    # data = request.data
    # nparr = np.fromstring(data.decode('base64'), np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


def encode_img(img):
    """
    response로 전송해줄 image를 encoding해주고 return하는 함수
    :param img: np.array
    :return: string
    """
    # byte_arr = io.BytesIO()
    # img.save(byte_arr, format='JPEG')
    # encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    encoded_img = cv2.imencode('.jpg', img).tostring()
    return encoded_img


if __name__ == '__main__':
    # media/video/box.mp4 -> frames save to media/image/

    video_name = 'box.mp4'
    path = FileControl()
    video_path = path.get_video_path(video_name)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    room = 0
    count = 0
    while success:
        image_path = path.get_image_path(room, count)
        cv2.imwrite(image_path, image)  # save frame as JPEG file
        print(image_path)
        success, image = vidcap.read()
        print('Read a new frame #{}: {}'.format(count, success))
        count += 1
