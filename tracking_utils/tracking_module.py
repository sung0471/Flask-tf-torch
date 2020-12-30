# ! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
변경사항
- 디렉토리 변경에 따른 import path 수정
- 트래킹영상.avi/디텍션결과.txt/tracking model들의 path 변경
"""

from __future__ import division, print_function, absolute_import
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from timeit import time
import warnings
import cv2
import numpy as np
# import argparse
from PIL import Image
# from yolo import YOLO

from tracking_utils.deep_sort import preprocessing, nn_matching
from tracking_utils.deep_sort import Detection
from tracking_utils.deep_sort import Tracker
from tracking_utils.tools import generate_detections as gdet
from tracking_utils.util import convert_to_origin_shape, load_class_name
# from deep_sort.detection import Detection as ddet
from tracking_utils.detect import model_detection, select_yolo
from tracking_utils.cfg import get_args_hyp
from collections import deque
# from keras import backend
import tensorflow as tf
# from tensorflow.compat.v1 import InteractiveSession
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", help="path to input video", default="./test_video/TownCentreXVID.avi")
# ap.add_argument("-c", "--class", help="name of class", default="person")
# args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
# list = [[] for _ in range(100)]


def main(location, frame, img):
    """
    tracking 해주는 함수
    현재는 기존의 tracking code가 그대로 임
    :param location: int
        img가 촬영된 장소
    :param frame: int
        img의 frame 번호(시간이 될 수 있음)
    :param img: np.array
        img array
    :return: str, str, np.array, int
        filepath, filename, img array, box count
    """
    args, hyp = get_args_hyp()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    start = time.time()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    counter = []
    # deep_sort
    model_filename = './tracking_utils/saved_model/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    find_objects = ['person']
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # select yolo
    YOLO, input_details, output_details, saved_model_loaded = select_yolo(args, hyp)

    write_video_flag = True
    video_capture = cv2.VideoCapture(args.input)

    # Define the codec and create VideoWriter object
    if write_video_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./media/output/output.avi', fourcc, 15, (w, h))
        list_file = open('./media/output/detection_rslt.txt', 'w')
        frame_index = -1

    fps = 0.0
    end_time = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if not ret:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])    # bgr to rgb
        # boxes, confidence, class_names = yolo.detect_image(image)

        image = np.squeeze(image)
        img = image.copy() / 255.0
        h, w, _ = img.shape
        if h != args.img_size or w != args.img_size:
            img = cv2.resize(img, (args.img_size, args.img_size))

        convert_class_name = load_class_name(args.data_root, args.class_file)

        # boxes, scores, classes, valid_detections = model_detection(img, YOLO, args,
        #                                                            input_details=input_details,
        #                                                            output_details=output_details)
        #
        # bbox_thick = int(0.6 * (h + w) / 600)
        # scores, classes, valid_detections = np.float32(np.squeeze(scores)), np.int32(np.squeeze(classes)), np.int32(
        #     np.squeeze(valid_detections))
        # fontScale = 0.5
        # box_color = (0, 255, 0)
        #
        # y_min, x_min, y_max, x_max = convert_to_origin_shape(boxes, None, None, h, w)
        # y_min, x_min, y_max, x_max = np.float32(np.reshape(y_min, -1)), np.float32(np.reshape(x_min, -1)), np.float32(
        #     np.reshape(y_max, -1)), np.float32(np.reshape(x_max, -1))
        # for i in range(valid_detections):
        #     if scores[i] < args.score_threshold:
        #         break
        #
        #     # draw rectangle
        #     image = cv2.rectangle(image, (x_min[i], y_min[i]), (x_max[i], y_max[i]), box_color, thickness=2)
        #
        #     # draw text
        #     bbox_mess = '%s: %.2f' % (convert_class_name[classes[i]], scores[i])
        #
        #     t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        #     t = (x_min[i] + t_size[0], y_min[i] - t_size[1] - 3)
        #     image = cv2.rectangle(image, (x_min[i], y_min[i]), (np.float32(t[0]), np.float32(t[1])), box_color,
        #                           -1)  # filled
        #     image = cv2.putText(image, bbox_mess, (int(x_min[i]), int(y_min[i] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
        #                         fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        #     # img = cv2.cvtColor(detect(img,YOLO,class_name,args,input_details,output_details),cv2.COLOR_RGB2BGR)
        #     cv2.imshow('detected',image)

        boxes, confidence, class_names, valid_detections = model_detection(img, YOLO, args, input_details, output_details)

        # print(boxes.shape)      # (100, 4)
        # print(class_names.shape)    # (100)
        # print(valid_detections.shape)   # (1,)
        # print(boxes, confidence, class_names, valid_detections)

        y_min, x_min, y_max, x_max = convert_to_origin_shape(boxes, None, None, h, w)
        # y_min, x_min, y_max, x_max = np.float32(np.reshape(y_min, -1)), np.float32(np.reshape(x_min, -1)), np.float32(
        #     np.reshape(y_max, -1)), np.float32(np.reshape(x_max, -1))
        boxes = np.concatenate([x_min, y_min, x_max - x_min, y_max - y_min], -1)

        boxes = tf.squeeze(boxes, 0)[:valid_detections[0]]      # 100, 4
        # confidence = tf.squeeze(confidence, 0)[:valid_detections[0]]
        # class_names = tf.squeeze(class_names, 0)

        features = encoder(frame, boxes)
        # score to 1.0 here.
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            # print(class_names)
            # print(class_names[p])

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            # print(frame_index)
            list_file.write(str(frame_index)+',')
            list_file.write(str(track.track_id)+',')
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            b0 = str(bbox[0])   # .split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1])   # .split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2]-bbox[0])   # .split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3]-bbox[1])

            list_file.write(str(b0) + ','+str(b1) + ','+str(b2) + ','+str(b3))
            # print(str(track.track_id))
            list_file.write('\n')
            # list_file.write(str(track.track_id)+',')
            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               cv2.putText(frame, str(convert_class_name[class_name[0].numpy()]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            # track_id[center]

            pts[track.track_id].append(center)

            thickness = 5
            # center point
            cv2.circle(frame,  (center), 1, color, thickness)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        count = len(set(counter))
        # cv2.putText(frame, "Total Pedestrian Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Box Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f / %fs"%(fps, end_time),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow("YOLO4_Deep_SORT", 0)
        cv2.resizeWindow('YOLO4_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO4_Deep_SORT', frame)
        print('infer time : {}'.format(end_time))

        if write_video_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        end_time = time.time()-t1
        fps = (fps + (1./end_time)) / 2
        out.write(frame)
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(args.input[43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")
    print("[INFO]: model_image_size = (960, 960)")
    video_capture.release()
    if write_video_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

    return '', '', img, count
