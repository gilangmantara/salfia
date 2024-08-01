# YOLOv3
"""
Run inference on images videos stream

Usage Yolo Weight:
    $ python salfia_detect.py --model yolo                                  # model name (yolo or ssd)
                              --label yolo                                  # labeling type (yolo or voc)
                              --name yolo                                   # output folder
                              --interval 1                                  # image delay timer
                              --num_img 10                                  # number of image to detect
                              --thresh 0.5                                  # confidence threshold
                              --weights weights/yolo_toll.pt                # model weights
                              --url https://url/path/reloaded_image_path    # image video url (use http:// or https://)

Usage SSD Weight:
    $ python salfia_detect.py --model ssd                                  # model name (yolo or ssd)
                              --label voc                                  # labeling type (yolo or voc)
                              --name ssd                                   # output folder
                              --interval 1                                  # crawling interval in seconds
                              --num_img 10                                  # number of image for crawling
                              --thresh 0.5                                  # confidence threshold
                              --weights weights/ssd_toll.pth                # model weights
                              --url https://url/path/reloaded_image_path    # image video url (use http:// or https://)
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, check_requirements, colorstr, increment_path, non_max_suppression,
                           print_args, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

# Crawling libs
import urllib.request
import cv2
import numpy as np
import datetime
import time

# SSD
from ssd.config import cfg
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer
from lxml.etree import Element, SubElement, tostring

# ROOT folder
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Classes to generate yolo/ssd class conversion
YOLO_CLASSES = ('1', '2', '3', '4', '5')


@torch.no_grad()
def run_yolo(url='',  # crawling URL
             interval=1,  # crawling interval in seconds
             num_img=5,  # number of image for crawling
             weights='weights/yolo_toll.pt',  # model.pt path(s)
             label='yolo',  # label type (yolo/voc)
             imgsz=640,  # inference size (pixels)
             conf_thres=0.5,  # confidence threshold
             iou_thres=0.45,  # NMS IOU threshold
             max_det=1000,  # maximum detections per image
             device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
             project=ROOT / 'output/',  # save results to project/name
             name='labeled',  # save results to project/name
             half=False,  # use FP16 half-precision inference
             dnn=False,  # use OpenCV DNN for ONNX inference
             ):
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # Crawling Image from URL
    for i in range(int(num_img)):
        # Get image
        url_response = urllib.request.urlopen(url)
        ts = datetime.datetime.now().timestamp()
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        filepath = os.path.join(save_dir, '{}.jpg'.format(ts))
        image_name = os.path.basename(filepath)

        # Write image to project/name folder
        cv2.imwrite(filepath, img)

        # Read image
        img0 = cv2.imread(filepath)  # BGR
        # Padded resize
        img = letterbox(img0, imgsz, stride, pt)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        t1 = time_sync()
        im = torch.from_numpy(img).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, None, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = img0.copy()

            s = 'image'
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                boxes = []
                labels = []

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format

                    xymin = tuple(np.copy(xyxy))
                    boxes.append(list(xymin))
                    labels.append(int('%g' % cls))

                boxes = np.array(boxes)
                labels = np.array(labels)

                if label == 'voc':
                    write_voc(img0, image_name, boxes, labels, save_dir)
                else:
                    write_yolo(img0, image_name, boxes, labels, save_dir)
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
        time.sleep(interval)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image' % t)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def write_yolo(image, image_name, boxes, labels, save_dir):
    gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for i in range(len(boxes)):
        xywh = (xyxy2xywh(torch.tensor(boxes[i]).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (labels[i], *xywh)  # label format
        with open(os.path.join(save_dir, os.path.splitext(image_name)[0] + '.txt'), 'a') as f:
            f.write(('%s ' * len(line)).rstrip() % line + '\n')


@torch.no_grad()
def run_ssd(url='',  # crawling URL
            interval=1,  # crawling interval in seconds
            num_img=5,  # number of image for crawling
            cfg=cfg,
            ckpt='weights/ssd_toll.pth',
            label='yolo',  # label type (yolo/voc)
            imgsz=640,  # inference size (pixels)
            score_threshold=0.7,
            device='',
            project=ROOT / 'output/',
            name='ssd',
            ):
    # Load model
    device = select_device(device)
    model = build_detection_model(cfg)
    stride = 64
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # Load Weights
    # checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    # checkpointer.load(ckpt, use_latest=ckpt is None)
    # weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    weight_file = ckpt
    print('Loaded weights from {}'.format(weight_file))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=False)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    cpu_device = device
    transforms = build_transforms(cfg, is_train=False)
    model.eval()

    for i in range(int(num_img)):
        seen += 1
        # Get image
        url_response = urllib.request.urlopen(url)
        ts = datetime.datetime.now().timestamp()
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        image_paths = os.path.join(save_dir, '{}.jpg'.format(ts))
        image_name = os.path.basename(image_paths)

        # Write image to project/name folder
        cv2.imwrite(image_paths, img)

        # Read image
        img0 = cv2.imread(image_paths)  # BGR
        # Padded resize
        img0 = letterbox(img0, imgsz, stride)[0]
        # Convert
        img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        image = np.array(img1)

        # SSD START
        start = time.time()
        # image_name = os.path.basename(image_paths)
        # image = np.array(Image.open(image_paths).convert("RGB"))

        height, width = image.shape[:2]
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = model(images.to(device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(cpu_device).numpy()
        boxes, labels, scores = result['boxes'], result['labels'], result['scores']

        indices = scores > score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        # scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

        # CONTINUE
        print(labels)
        labels = [YOLO_CLASSES.index(str(c)) for c in labels]
        if label == 'yolo':
            write_yolo(image, image_name, boxes, labels, save_dir)
        else:
            write_voc(image, image_name, boxes, labels, save_dir)

        time.sleep(interval)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image' % t)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def write_voc(image, image_name, boxes, labels, save_dir):
    height, width, channels = image.shape

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'XML'
    img_name = image_name

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'Toll database'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(channels)

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    xml = ""

    for j in range(len(boxes)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = YOLO_CLASSES[labels[j]]

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(boxes[j][0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(boxes[j][1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(boxes[j][2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(boxes[j][3]))
        xml = tostring(node_root, pretty_print=True)
    if xml:
        f = open(os.path.join(save_dir, os.path.splitext(img_name)[0] + '.xml', ), "wb")
        f.write(xml)
        f.close()
        print('write XML DONE!')


def parse_opt():
    parser = argparse.ArgumentParser()

    # Required Option
    parser.add_argument("--model", help="yolo or ssd")
    parser.add_argument("--label", help="label type (yolo or voc)")
    parser.add_argument("--url", help="Insert Your URL")

    # Additional Option
    parser.add_argument("--interval", default=1, type=int, help="Crawling Interval in second")
    parser.add_argument("--num_img", default=5, type=int, help="Number Of Image")
    parser.add_argument('--weights', type=str, default=None, help='model path(s)')
    parser.add_argument('--project', default=ROOT / 'output', help='save results to project/name')
    parser.add_argument('--name', default='labeled', help='save results to project/name')
    parser.add_argument('--thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument("--config-file", default="weights/ssd_toll.yaml", metavar="FILE", help="path to config file",
                        type=str)

    opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)

    if opt.model == 'ssd':
        # cfg_file = opt.config_file
        # cfg.merge_from_file(cfg_file)
        cfg.MODEL.NUM_CLASSES = 6
        cfg.freeze()
        opt.cfg = cfg

    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    print(opt.model)
    if opt.model == 'yolo':
        run_yolo(url=opt.url,
                 interval=opt.interval,
                 num_img=opt.num_img,
                 weights=opt.weights if opt.weights is not None else 'weights/yolo_toll.pt',
                 label=opt.label,
                 conf_thres=opt.thresh,
                 project=opt.project,
                 name=opt.name)
    elif opt.model == 'ssd':
        run_ssd(url=opt.url,
                interval=opt.interval,
                num_img=opt.num_img,
                cfg=opt.cfg,
                ckpt=opt.weights if opt.weights is not None else 'weights/ssd_toll.pth',
                label=opt.label,
                score_threshold=opt.thresh,
                project=opt.project,
                name=opt.name)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
