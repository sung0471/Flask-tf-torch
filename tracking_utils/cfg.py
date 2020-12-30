import argparse


def get_args_hyp():
    parser = argparse.ArgumentParser(description='YOLOv4 Test')
    parser.add_argument("-i", "--input", help="path to input video", default="./media/video/box.mp4")
    parser.add_argument("-c", "--class", help="name of class", default="person")
    parser.add_argument('--img_size',              type=int,   help='Size of input image / default : 416', default=416)
    parser.add_argument('--data_root',              type=str,   help='Root path of class name file and coco_%2017.txt / default : "tracking/data"', default='./data')
    parser.add_argument('--class_file',              type=str,   help='Class name file / default : "coco.name"', default='box.names')   # 'coco.names'
    parser.add_argument('--num_classes', type=int, help='Number of classes (in COCO 80) ', default=80)  # 80
    parser.add_argument('--weight_path', type=str ,default='./tracking_utils/saved_model/box', help='path of weight')    # 'dark_weight/yolov4.weights'
    parser.add_argument('--is_saved_model', action='store_true', help='If ture, load saved model', default=True)
    parser.add_argument('--is_tflite', action='store_true', help='If ture, load saved model', default=False)
    parser.add_argument('--is_darknet_weight', action='store_true', help = 'If true, load the weight file used by the darknet framework.')  # 'store_false'
    parser.add_argument('--is_tiny', action='store_true', help='Flag for using tiny. / default : false')
    parser.add_argument('--input_dir', type=str, default='./data/dataset/box_dataset/media/val')
    parser.add_argument('--confidence_threshold', type=float, default=0.001)
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument('--data_name', type=str,
                        help='Root path of class name file and coco_%2017.txt / default : "./data"'
                        , default='coco')
    args = parser.parse_args()

    args.mode ='eval'
    args.soft = 0.0
    args.batch_size = 1

    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'iou_t': 0.213,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0005,  # final learning rate (with cos scheduler)
           'momentum': 0.949,  # SGD momentum
           'weight_decay': 0.000484,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.5,  # image scale (+/- gain)
           'shear': 0.641 * 0}  # image shear (+/- deg)

    return args, hyp
