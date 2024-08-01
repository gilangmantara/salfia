## Yolo 3 Semi-Auto Labeling
Original Yolo v3 from [Ultralytics YOLOv3 GitHub](https://github.com/ultralytics/yolov3/) and modified for Collect **Toll Road Classification Dataset** using Semi-Auto Labeling

## About Weights
Pretrained weight for Indonesian Toll Road Classification: [Download Weights](/weights/)

## First-Time Setup
Clone repo and install [requirements.txt](/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://git.brin.go.id/gila003/salfia  # clone
cd yolov3
pip install -r requirements.txt  # install
```

## Usage
### Required Options
- **--model**

  model architecture for prediction.
  Can be one of:
  - **yolo**: predict with yolo architecture. 
  - **ssd**: predict with ssd architecture.

- **--label**
    
  Image label format for the result.
  Can be one of:
  - **yolo**: yolo for txt format.
  - **voc**: voc for xml format.

- **--url**

  image video url (use **http://** or **https://**)

### Additional Options
- **--name**
    
  Output folder for the result.

- **--interval**
    
  delay timer for processing image labeling

- **--num_img**

  number of total image to process

- **--thresh**

  confidence threshold (0.0 - 1.0), default 0.5

- **--weights**

  model weights

- **--device**

  compute engine.
  Can be one of:
  - **cpu**: compute with CPU. 
  - **gpu**: compute with GPU.

### Usage Yolo Weight
replace **your_image_video_url** with image url (use **http://** or **https://**)
#### simple option
```bash
python salfia_detect.py --model yolo --label yolo --url https://your_image_video_url
```
#### full option
```bash
python salfia_detect.py --model yolo --label yolo --url https://your_image_video_url --name yolo --interval 1 --num_img 10 --thresh 0.5 --weights weights/yolo_toll.pt
```

### Usage SSD Weight
replace **your_image_video_url** with image url (use **http://** or **https://**)
#### simple option
```bash
python salfia_detect.py --model ssd --label voc --url https://your_image_video_url
```
#### full option
```bash
python salfia_detect.py --model ssd --label voc --url https://your_image_video_url --name ssd --interval 1 --num_img 10 --thresh 0.5 --weights weights/ssd_toll.pth
```
