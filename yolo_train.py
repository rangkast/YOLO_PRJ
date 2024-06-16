import os
import json
import glob
import shutil
import random
from ultralytics import YOLO
import torch
import cv2

def convert_to_yolo_format(label_file, output_dir, img_width, img_height):
    with open(label_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for item in data['images']:  # 모든 이미지 사용
        image_path = item['file']
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(label_path, 'w') as lf:
            for ann in item['annotations']:
                label = int(ann['label'])
                if label == -1:
                    continue
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2]) / 2.0 / img_width
                y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height
                lf.write(f"{label} {x_center} {y_center} {width} {height}\n")

def prepare_dataset(image_dir, label_dir, output_dir):
    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')

    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)

    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))  # 모든 이미지 사용
    for img_path in image_files:
        img = cv2.imread(img_path)  # RGB 이미지로 읽기
        img_resized = cv2.resize(img, (640, 640))
        img_output_path = os.path.join(img_output_dir, os.path.basename(img_path))
        cv2.imwrite(img_output_path, img_resized)

        lbl_path = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        src_lbl_path = os.path.join(label_dir, lbl_path)
        dst_lbl_path = os.path.join(lbl_output_dir, lbl_path)
        if src_lbl_path != dst_lbl_path:
            shutil.copy(src_lbl_path, dst_lbl_path)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = f"{script_dir}/images"
    label_file = f"{script_dir}/NewTrainingRaw_labels_updated.json"  # 업데이트된 JSON 파일 사용
    output_dir = f"{script_dir}/yolo_dataset"
    
    label_dir = os.path.join(output_dir, 'labels')
    img_width, img_height = 1920, 1080  # 원본 이미지의 크기

    convert_to_yolo_format(label_file, label_dir, img_width, img_height)
    prepare_dataset(image_dir, label_dir, output_dir)

    data_yaml = f"""
    path: {output_dir}
    train: images
    val: images
    nc: 10
    names: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    """
    
    with open(f"{output_dir}/data.yaml", 'w') as f:
        f.write(data_yaml)

    # GPU 메모리 캐시 초기화 및 환경 변수 설정
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    
    model = YOLO('yolov8s.pt')

    # 학습 시작 여기 성능을 늘리면 뻗음 ㅠㅠ
    model.train(data=f"{output_dir}/data.yaml", epochs=100, batch=2, imgsz=640)
