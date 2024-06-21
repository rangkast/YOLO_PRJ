import os
import cv2
import glob
import json
import shutil
from ultralytics import YOLO
import torch
import numpy as np

def image_pyramid(image, scale=1.5, min_size=(30, 30)):
    yield image
    count = 0
    while count < 1:
        width = int(image.shape[1] / scale)
        height = int(image.shape[0] / scale)
        if width < min_size[0] or height < min_size[1]:
            break
        image = cv2.resize(image, (width, height))
        yield image
        count += 1

def adjust_bboxes(bboxes, img_width, img_height):
    adjusted_bboxes = []
    for bbox in bboxes:
        label, x_center, y_center, width, height = bbox
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        adjusted_bboxes.append((label, x_center, y_center, width, height))
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
                x1, y1, x2, y2 = ann['bbox']
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height
                lf.write(f"{label} {x_center} {y_center} {width} {height}\n")

def rotate_bbox(bbox, angle, img_width, img_height):
    label, x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    cx, cy = img_width / 2, img_height / 2
    angle_rad = np.deg2rad(angle)

    corners = np.array([
        [x_center - width / 2, y_center - height / 2],
        [x_center + width / 2, y_center - height / 2],
        [x_center + width / 2, y_center + height / 2],
        [x_center - width / 2, y_center + height / 2]
    ])

    new_corners = []
    for corner in corners:
        x_shifted, y_shifted = corner[0] - cx, corner[1] - cy
        new_x = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad) + cx
        new_y = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad) + cy
        new_corners.append([new_x, new_y])

    new_corners = np.array(new_corners)
    x_min, y_min = np.min(new_corners, axis=0)
    x_max, y_max = np.max(new_corners, axis=0)

    new_x_center = (x_min + x_max) / 2.0 / img_width
    new_y_center = (y_min + y_max) / 2.0 / img_height
    new_width = (x_max - x_min) / img_width
    new_height = (y_max - y_min) / img_height

    return (label, new_x_center, new_y_center, new_width, new_height)

def augment_image(image, bboxes, img_width, img_height):
    augmented_images = []
    augmented_bboxes = []

    # 회전 증강
    angles = [-15, 15]
    for angle in angles:
        M = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (img_width, img_height))

        new_bboxes = []
        for bbox in bboxes:
            new_bbox = rotate_bbox(bbox, angle, img_width, img_height)
            new_bboxes.append(new_bbox)

        augmented_images.append(rotated_image)
        augmented_bboxes.append(new_bboxes)

    # 크기 증강
    scales = [0.8, 1.2]
    for scale in scales:
        scaled_image = cv2.resize(image, None, fx=scale, fy=scale)

        new_bboxes = []
        for bbox in bboxes:
            label, x_center, y_center, width, height = bbox
            new_bbox = (label, x_center / scale, y_center / scale, width / scale, height / scale)
            new_bboxes.append(new_bbox)

        augmented_images.append(scaled_image)
        augmented_bboxes.append(new_bboxes)

    # 밝기 증강
    bright_values = [0.8, 1.2]
    for bright in bright_values:
        bright_image = cv2.convertScaleAbs(image, alpha=bright, beta=0)

        augmented_images.append(bright_image)
        augmented_bboxes.append(bboxes)

    # 노이즈 증강
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    augmented_images.append(noisy_image)
    augmented_bboxes.append(bboxes)

    # 블러 증강
    ksize = (5, 5)
    blurred_image = cv2.GaussianBlur(image, ksize, 0)

    augmented_images.append(blurred_image)
    augmented_bboxes.append(bboxes)

    return augmented_images, augmented_bboxes

def prepare_dataset(image_dir, label_dir, output_dir, debug_dir, img_width=640, img_height=640):
    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))
    for img_path in image_files[:1]:  # 한 장의 이미지만 처리
        img = cv2.imread(img_path)
        original_height, original_width = img.shape[:2]

        lbl_path = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        src_lbl_path = os.path.join(label_dir, lbl_path)
        dst_lbl_path = os.path.join(lbl_output_dir, os.path.basename(lbl_path))

        if os.path.exists(src_lbl_path):
            with open(src_lbl_path, 'r') as f:
                bboxes = [list(map(float, line.strip().split())) for line in f]
            bboxes = adjust_bboxes(bboxes, original_width, original_height)

            # 데이터 증강
            augmented_images, augmented_bboxes = augment_image(img, bboxes, img_width, img_height)
            for i, (aug_img, aug_bboxes) in enumerate(zip(augmented_images, augmented_bboxes)):
                img_resized = cv2.resize(aug_img, (img_width, img_height))
                img_output_path = os.path.join(debug_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{i}.jpg")
                cv2.imwrite(img_output_path, img_resized)

                for bbox in aug_bboxes:
                    label, x_center, y_center, width, height = bbox
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)
                    cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_resized, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                debug_output_path = os.path.join(debug_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{i}_bbox.jpg")
                cv2.imwrite(debug_output_path, img_resized)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    original_image_dir = f"{script_dir}/images"
    new_image_dir = f"{script_dir}/images_01"
    original_label_file = f"{script_dir}/NewTrainingRaw_labels_updated.json"
    new_label_file = f"{script_dir}/labels.json"
    original_output_dir = f"{script_dir}/yolo_dataset"
    new_output_dir = f"{script_dir}/yolo_dataset_01"
    merged_output_dir = f"{script_dir}/yolo_dataset_merged"
    debug_dir = f"{script_dir}/debug"

    label_dir = os.path.join(original_output_dir, 'labels')
    img_width, img_height = 1920, 1080

    # 기존 데이터셋 준비
    convert_to_yolo_format(original_label_file, label_dir, img_width, img_height)
    prepare_dataset(original_image_dir, label_dir, original_output_dir, debug_dir)
