# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings
import cv2
import numpy as np
import datetime as dt
from tracking_utils.deep_sort import preprocessing, nn_matching
from tracking_utils.deep_sort import Detection
from tracking_utils.deep_sort import Tracker
from tracking_utils.tools import generate_detections as gdet
from tracking_utils.util import convert_to_origin_shape, load_class_name
from tracking_utils.detect import model_detection, select_yolo
from tracking_utils.cfg import get_args_hyp
from collections import deque
import tensorflow as tf
warnings.filterwarnings('ignore')
np.random.seed(100)

# default setting
args = get_args_hyp()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# select yolo
YOLO, input_details, output_details, saved_model_loaded = select_yolo(args)

# load class name information
convert_class_name = load_class_name(args.data_root, args.class_file)

# deep_sort model, hyp
model_filename = './tracking_utils/saved_model/market1501.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
max_cosine_distance = 0.3
nms_max_overlap = 1.0

# tracker metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, None)

class LocationInfo:
    def __init__(self):
        self.tracker = dict()
        self.colors = dict()
        self.last_save_time = dict()
        self.last_box = dict()

    def check(self,location):
        if location not in self.tracker.keys():
            self.tracker[location] = Tracker(metric)
            self.colors[location] = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
            self.last_box[location] = 0
            self.last_save_time[location] = None

    def get(self,location):
        return self.tracker[location],self.colors[location]

    def time_check(self,location,time):
        times = [int(t) for t in time.split('_')]
        times[-1] *=int(1e4)
        now_time = dt.datetime(*times)
        
        if self.last_save_time[location]==None or (now_time - self.last_save_time[location]).total_seconds() >= 30:
            self.last_save_time[location] = now_time
            return True
        else:
            return False

location_info = LocationInfo()

def main(location, time, image):
    """
    :param
        location: int
        Location number of input image
    :param
        time: int
        date time of input image [format : yyyy_mm_dd_hh-mm_ss_ss]
    :param
        image: np.array
        image drawed box

    :return: 
        write_path : str
            Path of saved image
        image : np.array
            image drawed box
        box_count : int
            counted number of box
    """

    # load tracker for specific location
    location_info.check(location)
    tracker,COLORS = location_info.get(location)
    is_save = location_info.time_check(location,time)

    # image preprocessing
    image = np.squeeze(image)
    img = image.copy() / 255.0
    h, w, _ = img.shape
    if h != args.img_size or w != args.img_size:
        img = cv2.resize(img, (args.img_size, args.img_size))

    # get bounding box from YOLO
    boxes, confidence, class_names, valid_detections = model_detection(img, YOLO, args, input_details, output_details)
    y_min, x_min, y_max, x_max = convert_to_origin_shape(boxes, None, None, h, w)
    boxes = np.concatenate([x_min, y_min, x_max - x_min, y_max - y_min], -1)
    boxes = tf.squeeze(boxes, 0)[:valid_detections[0]]      # 100, 4

    # extract feature from image patch
    features = encoder(image, boxes)

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

    box_count = int(0)

    # draw detected bounding box
    if is_save:
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

    # draw tracked bounding box
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        if is_save:
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[int(track.track_id) % len(COLORS)]]

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            b0 = str(bbox[0])   # .split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1])   # .split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2]-bbox[0])   # .split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3]-bbox[1])

            cv2.putText(image,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
                class_name = class_names[0]
                cv2.putText(image, str(convert_class_name[class_name[0].numpy()]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

        box_count += 1
    write_path = None
    if is_save or location_info.last_box[location] != box_count:
        pre_path = os.path.join(args.write_path,f'{location}')
        if not os.path.exists(pre_path):
            os.makedirs(pre_path)
        location_info.last_box[location] = box_count
        f = open(pre_path+f'/box_{time[:10]}.csv','w' if not os.path.exists(pre_path+f'/box_{time[:10]}.csv') else 'a')
        f.writelines(f'{time},{box_count}\n')

    if is_save:
        cv2.putText(image, "Current Box Counter: "+str(box_count),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        write_path = os.path.join(pre_path,f'{time}.jpg')
        cv2.imwrite(write_path,image)
    
    return write_path, image, box_count