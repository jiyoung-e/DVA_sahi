import argparse
import sys
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

# 패키지의 루트 경로를 sys.path에 추가합니다.
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
# from sahi.models.yolox import YoloxDetectionModel
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

import numpy as np


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "--model", default="yolov8", help="Model name | yolov5, yolox, yolov8"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--output_path",
        default="./Bytetrack_Outputs",
        help="Output Path",
    )
    parser.add_argument(
        "--sliced_size",
        default=1024,
        help="sliced width, height size",
    )
    parser.add_argument(
        "--overlap_ratio",
        default=0.2,
        help="Slice overlap ratio",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="If you want to use yolox, pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.2, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

class Predictor(object):
    def __init__(
        self,
        det_model,
        sliced_size = 1024,
        overlap_ratio = 0.2,
        device=torch.device("cuda:0"),
        fp16=False
    ):
        self.det_model = det_model
        
        self.device = device
        self.fp16 = fp16

        self.sliced_size = sliced_size
        self.overlap_ratio = overlap_ratio
        
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        img_path = img
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        file_name_without_extension, _ = os.path.splitext(os.path.basename(img_path))
        result = get_sliced_prediction(img_path, self.det_model, output_file_name=file_name_without_extension, 
                                       slice_height=self.sliced_size, slice_width=self.sliced_size, overlap_width_ratio=self.overlap_ratio, overlap_height_ratio=self.overlap_ratio, postprocess_type="NMS")
    
        outputs = []
        for ann in result.to_coco_annotations(): 
            bbox = ann['bbox']
            conf = ann['score']
            label = ann['category_id']
            # output : [x1, y1, x2, y2, conf, label]
            outputs.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], conf, label])
        outputs = torch.Tensor(outputs)
        
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []
    
    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        
        if outputs is not None and np.array(outputs).size > 0:
            online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_labels = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                label = int(t.label)
                
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_labels.append(label)
                    # save results
                    results.append(f"{frame_id},{tid},{label},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")
            timer.toc()
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, online_labels, frame_id=frame_id + 1, fps=1. / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main():
    args = make_parser().parse_args()

    vis_folder = osp.join(args.output_path, args.model)
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))
    logger.info("Selected Model: ", args.model)

    if args.model == "yolox":
        exp = get_exp(args.exp_file, args.name)

        if not args.experiment_name:
            args.experiment_name = exp.exp_name

        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)

    # Model define
    if args.model == 'yolov5':
        detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov5',
        confidence_threshold=0.3,
        image_size = 1024,
        model_path = args.ckpt ,
        device='cuda:0', # or 'cpu'
        )
    elif args.model == 'yolov8':
        detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        confidence_threshold=0.3,
        image_size = 1024,
        model_path= args.ckpt,
        device='cuda:0', # or 'cpu'
    )
    predictor = Predictor(detection_model, args.sliced_size, args.overlap_ratio, args.device, args.fp16)

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    else:
        logger.exception("Please check input format")


if __name__ == "__main__":
    main()
