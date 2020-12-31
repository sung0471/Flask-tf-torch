from PIL import Image
from flask import Flask, make_response, send_from_directory
from flask_restful import Resource, Api
from utils.utils import get_param_parsing, get_image, encode_img, FileControl
from tracking_utils.tracking_module import main as tracking

path = FileControl()

UPLOAD_FOLDER = path.image_dir
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)


class Index(Resource):
    description = '<h1>API Description</h1>'\
                  '<table border="1" style="border-collapse:collapse">'\
                  '<tr><td>/index</td><td>GET</td><td>API 설명 페이지(this)</td></tr>'\
                  '<tr><td>/tracking</td><td>POST</td><td>image를 전송해 tracking 처리하여 저장하는 API</td></tr>'\
                  '<tr><td>/get_image</td><td>GET</td><td>Image를 요청하여 받는 API</td></tr>'\
                  '</table>'

    def get(self):
        res = make_response(self.description)
        res.headers['Content-type'] = 'text/html; charset=utf-8'
        return res

class TrackingImage(Resource):
    """
    :description
        Image를 HTTP post 방식으로 받아서 tracking해주는 API
        location(방 번호)와 frame(프레임 번호)는 GET 방식으로 입력받음
    :return
        JSON
        :key
            box_num(int)
            image(binary string)
        :value
            box number per image(int)
            image data(binary string)
    """
    def get(self):
        return self.post()

    @staticmethod
    def post(): # client
       
        location, frame_num, _ = get_param_parsing()
        print('localtion',location)
        img = get_image()
        # img.show()
        # print(type(img))
        
        file_path, img, box_num = tracking(location, frame_num, img) # 저장한 위치,
        # ./media/tracking/{room_number}/{fil_name}     # str
        # img class = numpy object
        # box_num                                                          # result / int
        
        encoded_img = encode_img(img)

        res = make_response({'file_path':file_path,'box_num':box_num})
        #res = make_response({'box_num': box_num, 'image': encoded_img})
        res.headers['Content-type'] = 'application/json'

        return res


class SendImage(Resource):
    """특정 location, time에 해당하는 사진 전송
    
    Returns:
        Json
            box_num ([int]): 해당 이미지에서 검출된 box 개수
            image ([binary string]): 검출 결과가 그려진 이미지
    """
    
    @staticmethod
    def get():
        location, time, show_image = get_param_parsing()
        
        # return time, saved_image, box num
        if show_image:
            file_path, file_name = path.get_tracked_image_path(location, time, return_join=False)

            return send_from_directory(file_path, file_name)
        else:
            img_path = path.get_tracked_image_path(location, time)
            img = Image.open(img_path, mode='r')
            encoded_img = encode_img(img)

            res = make_response({'image': encoded_img, 'box_num': 0})
            res.headers['Content-type'] = 'application/json'

            return res

class SendInfo(Resource):
    """특정 location에 대한 time_list 반환

    Returns:
        ([str list]): 특정 location에 대한 time_list
    """
    @staticmethod
    def get():
        location = get_param_parsing()
        res = make_response({'time_list': get_time_list(location)})
        res.headers['Content-type'] = 'application/json'
        return res

api.add_resource(Index, '/', '/index')
# API를 간단히 설명해주는 페이지
api.add_resource(TrackingImage, '/tracking')
# /tracking?location=a&frame=b
# a = 방의 위치(카메라의 번호), default=0
# b = 보내주는 프레임의 번호(나중에 시간으로 바뀔 수 있음 or 없어질 수 있음), default=0
api.add_resource(SendImage, '/get_image')
api.add_resource(SendInfo, './location_info')

# /get_image?location=a&frame=b&show_image=False
# a = 방의 위치(카메라의 번호), default=0
# b = 프레임 번호(나중에 시간으로 바뀔 수 있음 or 없어질 수 있음), default=0
# imageonly = 이미지 자체로 단독으로 받을지, binary data로 다른 데이터와 같이 받을지 선택, default=False


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run()
