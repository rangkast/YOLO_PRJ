import os
import json
import glob
import shutil
import random
from ultralytics import YOLO
import torch
import cv2
import numpy as np

def adjust_contrast(image, alpha=1.5):
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return new_image

def adjust_color(image, hue_shift=10, saturation_scale=1.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv_image)
    hue = cv2.add(hue, hue_shift)
    saturation = cv2.convertScaleAbs(saturation, alpha=saturation_scale, beta=0)
    hsv_image = cv2.merge([hue, saturation, value])
    new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return new_image

def add_noise(image, noise_factor=0.005):
    noise = np.random.randn(*image.shape) * 255 * noise_factor
    noisy_image = image + noise.astype(np.float32)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def apply_gaussian_blur(image, kernel_size=(3, 3), sigma=0.5):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def apply_sharpening_filter(image, alpha=0.2):
    kernel = np.array([[0, -1, 0], 
                       [-1, 4 * alpha + 1, -1], 
                       [0, -1, 0]])  # 약한 샤프니스 필터
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def convert_to_yolo_format(label_file, output_dir, img_width, img_height):
    with open(label_file, 'r') as f:
        data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
    labeling_ten = []
    for item in data['images']:
        if len(item['annotations']) == 10:
            image_path = item['file']
            labeling_ten.append(image_path)
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
    return labeling_ten

def prepare_dataset(image_dir, label_dir, output_dir, labeling, contrast=True, color=True, noise=True, blur=True, sharpen=False):
    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)
    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))
    for img_path in image_files:
        detect_status = False
        for labeling_data in labeling:
            if labeling_data in img_path:
                detect_status = True
                break

        if detect_status:
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (640, 640))
            img_output_path = os.path.join(img_output_dir, os.path.basename(img_path))
            cv2.imwrite(img_output_path, img_resized)
            
            lbl_path = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            src_lbl_path = os.path.join(label_dir, lbl_path)
            dst_lbl_path = os.path.join(lbl_output_dir, lbl_path)
            if src_lbl_path != dst_lbl_path:
                shutil.copy(src_lbl_path, dst_lbl_path)
            
            # 동일한 라벨 파일을 증강 이미지에 대해 생성
            if contrast:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_contrast.txt"))
            if color:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_color.txt"))
            if noise:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_noisy.txt"))
            if blur:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_blur.txt"))
            if sharpen:
                shutil.copy(dst_lbl_path, os.path.join(lbl_output_dir, f"{os.path.splitext(os.path.basename(lbl_path))[0]}_sharpen.txt"))

            # 이미지 증강
            if contrast:
                img_contrast = adjust_contrast(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_contrast.jpg"), img_contrast)
            if color:
                img_color = adjust_color(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_color.jpg"), img_color)
            if noise:
                img_noisy = add_noise(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_noisy.jpg"), img_noisy)
            if blur:
                img_blur = apply_gaussian_blur(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_blur.jpg"), img_blur)
            if sharpen:
                img_sharpen = apply_sharpening_filter(img_resized)
                cv2.imwrite(os.path.join(img_output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_sharpen.jpg"), img_sharpen)

def merge_datasets(original_dir, new_dir, merged_dir):
    for subdir in ['images', 'labels']:
        os.makedirs(os.path.join(merged_dir, subdir), exist_ok=True)

        for file in glob.glob(os.path.join(original_dir, subdir, '*')):
            dst = os.path.join(merged_dir, subdir, os.path.basename(file))
            if os.path.abspath(file) != os.path.abspath(dst):
                shutil.copy(file, dst)

        for file in glob.glob(os.path.join(new_dir, subdir, '*')):
            dst = os.path.join(merged_dir, subdir, os.path.basename(file))
            if os.path.abspath(file) != os.path.abspath(dst):
                shutil.copy(file, dst)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    img_width, img_height = 1920, 1080  
    
    # dataset_1   
    dataset_1_image_dir = f"{script_dir}/images_1"
    dataset_1_label_file = f"{script_dir}/labels_1.json"
    dataset_1_output_dir = f"{script_dir}/yolo_dataset_1"
    label_dir = os.path.join(dataset_1_output_dir, 'labels')    
    labeling = convert_to_yolo_format(dataset_1_label_file, label_dir, img_width, img_height)
    prepare_dataset(dataset_1_image_dir, label_dir, dataset_1_output_dir, labeling)
    
    # dataset_2
    dataset_2_image_dir = f"{script_dir}/images_2"    
    dataset_2_label_file = f"{script_dir}/labels_2.json"    
    dataset_2_output_dir = f"{script_dir}/yolo_dataset_2"
    label_dir = os.path.join(dataset_2_output_dir, 'labels')    
    labeling = convert_to_yolo_format(dataset_2_label_file, label_dir, img_width, img_height)
    prepare_dataset(dataset_2_image_dir, label_dir, dataset_2_output_dir, labeling)
      
    # 데이터셋 병합
    merged_output_dir = f"{script_dir}/yolo_dataset_merged"    
    merge_datasets(dataset_1_output_dir, dataset_2_output_dir, merged_output_dir)
    
    data_yaml = f"""
    path: {merged_output_dir}
    train: images
    val: images
    nc: 10
    names: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    """
    
    with open(f"{merged_output_dir}/data.yaml", 'w') as f:
        f.write(data_yaml)
    
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    
    model = YOLO('yolov8s.pt')

    # 학습 시작
    model.train(data=f"{merged_output_dir}/data.yaml", epochs=50, batch=4, imgsz=640)
