import argparse


def get_args_hyp():
    parser = argparse.ArgumentParser(description='YOLOv4 Test')
    parser.add_argument('--img_size',              type=int,   help='Size of input image / default : 416', default=416)
    parser.add_argument('--data_root',              type=str,   help='Root path of class name file and coco_%2017.txt / default : "tracking/data"', default='./data')
    parser.add_argument('--class_file',              type=str,   help='Class name file / default : "coco.name"', default='box.names')   # 'coco.names'
    parser.add_argument('--weight_path', type=str ,default='./tracking_utils/saved_model/box', help='path of weight')    # 'dark_weight/yolov4.weights'
    parser.add_argument('--is_saved_model', action='store_true', help='If ture, load saved model', default=True)
    parser.add_argument('--is_tflite', action='store_true', help='If ture, load saved model', default=False)
    parser.add_argument('--is_darknet_weight', action='store_true', help = 'If true, load the weight file used by the darknet framework.')  # 'store_false'
    parser.add_argument('--is_tiny', action='store_true', help='Flag for using tiny. / default : false')
    parser.add_argument('--write_path', type=str, help = 'Path to save image',default = './media/tracking/' )
    parser.add_argument('--confidence_threshold', type=float, default=0.001)
    parser.add_argument('--iou_threshold', type=float, default=0.1)
    parser.add_argument('--score_threshold', type=float, default=0.3)
    args = parser.parse_args()

    args.mode ='eval'
    args.soft = 0.0
    args.batch_size = 1

    return args
