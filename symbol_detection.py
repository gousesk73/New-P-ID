import os
import sys
from pathlib import Path
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, increment_path, non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device, smart_inference_mode

import csv

@smart_inference_mode()
def run(
    weights=Path("yolov5s.pt"),  # model path
    source=Path("data/images"),  # file/dir
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_csv=True,  # save results to CSV file
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    project=Path("runs/detect"),  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False  # existing project/name ok, do not increment
):
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize CSV file
    csv_file = save_dir / "detections.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "class", "confidence", "x_center", "y_center", "width", "height"])


    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=None)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # normalize to 0 - 1.0
        if len(img.shape) == 3:  # if image does not have batch size
            img = img.unsqueeze(0)  # add batch dimension
        img = img.permute(0, 3, 1, 2)  # add batch dimension
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Write results to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        x_center, y_center, width, height = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                        writer.writerow([Path(path).name, names[int(cls)], conf, x_center, y_center, width, height])

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model path')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir with images')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    opt = parser.parse_args()
    run(**vars(opt))