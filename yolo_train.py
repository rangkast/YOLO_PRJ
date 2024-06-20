import cv2
import os
import glob
import json
import shutil
from ultralytics import YOLO
import torch

def image_pyramid(image, scale=1.5, min_size=(30, 30)):
    yield image
    count = 0
    while count < 2:  # 두 번만 스케일을 조정합니다.
        width = int(image.shape[1] / scale)
        height = int(image.shape[0] / scale)
        if width < min_size[0] or height < min_size[1]:
            break
        image = cv2.resize(image, (width, height))
        yield image
        count += 1

def adjust_bboxes(bboxes, scale):
    adjusted_bboxes = []
    for bbox in bboxes:
        label, x_center, y_center, width, height = bbox
        x_center /= scale
        y_center /= scale
        width /= scale
        height /= scale
        adjusted_bboxes.append((int(label), x_center, y_center, width, height))  # 라벨 값을 정수로 유지
    return adjusted_bboxes

def convert_to_yolo_format(label_file, output_dir, img_width, img_height):
    with open(label_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for item in data['images']:
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

def adjust_bboxes_for_resized_image(bboxes, original_width, original_height, resized_width, resized_height):
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    adjusted_bboxes = []
    for bbox in bboxes:
        label, x_center, y_center, width, height = bbox
        x_center /= scale_x
        y_center /= scale_y
        width /= scale_x
        height /= scale_y
        adjusted_bboxes.append((int(label), x_center, y_center, width, height))  # 라벨 값을 정수로 유지
    return adjusted_bboxes

def prepare_dataset(image_dir, label_dir, output_dir):
    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')

    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)

    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))
    for img_path in image_files:
        img = cv2.imread(img_path)
        original_height, original_width = img.shape[:2]

        # 원본 이미지를 640으로 리사이즈하여 저장
        img_resized = cv2.resize(img, (640, 640))
        img_output_path = os.path.join(img_output_dir, os.path.basename(img_path))
        cv2.imwrite(img_output_path, img_resized)

        lbl_path = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        src_lbl_path = os.path.join(label_dir, lbl_path)
        dst_lbl_path = os.path.join(lbl_output_dir, os.path.basename(lbl_path))

        if os.path.exists(src_lbl_path) and os.path.abspath(src_lbl_path) != os.path.abspath(dst_lbl_path):
            # 바운딩 박스를 원본 이미지의 크기에서 리사이즈된 이미지의 크기로 변환
            with open(src_lbl_path, 'r') as f:
                bboxes = [list(map(float, line.strip().split())) for line in f]
            adjusted_bboxes = adjust_bboxes_for_resized_image(bboxes, original_width, original_height, 640, 640)

            with open(dst_lbl_path, 'w') as f:
                for bbox in adjusted_bboxes:
                    label, x_center, y_center, width, height = bbox
                    f.write(f"{label} {x_center} {y_center} {width} {height}\n")

        # 스케일된 이미지를 저장
        for i, scaled_img in enumerate(image_pyramid(img)):
            img_resized = cv2.resize(scaled_img, (640, 640))
            scale_factor = original_width / scaled_img.shape[1]

            img_output_path = os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_scale_{i}.jpg")
            cv2.imwrite(img_output_path, img_resized)

            scaled_lbl_path = os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_scale_{i}.txt")
            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, 'r') as f:
                    bboxes = [list(map(float, line.strip().split())) for line in f]
                adjusted_bboxes = adjust_bboxes(bboxes, scale_factor)

                with open(scaled_lbl_path, 'w') as f:
                    for bbox in adjusted_bboxes:
                        label, x_center, y_center, width, height = bbox
                        f.write(f"{label} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = f"{script_dir}/images_01"
    label_file = f"{script_dir}/labels.json"
    output_dir = f"{script_dir}/yolo_dataset"
    
    label_dir = os.path.join(output_dir, 'labels')
    img_width, img_height = 1920, 1080

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

    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    
    model = YOLO('yolov8s.pt')
    model.train(data=f"{output_dir}/data.yaml", epochs=10, batch=2, imgsz=640)
