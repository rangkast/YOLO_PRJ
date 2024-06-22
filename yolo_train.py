import os
import cv2
import glob
import json
import shutil
from ultralytics import YOLO
import torch

def image_pyramid(image, scale=1.5, min_size=(30, 30)):
    yield image
    count = 0
    while count < 0:  # 두 번만 스케일을 조정합니다.
        width = int(image.shape[1] / scale)
        height = int(image.shape[0] / scale)
        if width < min_size[0] or height < min_size[1]:
            break
        image = cv2.resize(image, (width, height))
        yield image
        count += 1

def adjust_bboxes(bboxes, scale_x, scale_y):
    adjusted_bboxes = []
    for bbox in bboxes:
        label, x_center, y_center, width, height = bbox
        x_center /= scale_x
        y_center /= scale_y
        width /= scale_x
        height /= scale_y
        adjusted_bboxes.append((int(label), x_center, y_center, width, height))
    return adjusted_bboxes

def convert_to_yolo_format(label_file, output_dir, img_width, img_height):
    with open(label_file, 'r') as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for item in data['images']:
        image_path = item['file']
        if len(item['annotations']) != 10:
            continue

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
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    adjusted_bboxes = []
    for bbox in bboxes:
        label, x_center, y_center, width, height = bbox
        x_center *= scale_x
        y_center *= scale_y
        width *= scale_x
        height *= scale_y
        adjusted_bboxes.append((int(label), x_center, y_center, width, height))
    return adjusted_bboxes

def save_debug_image(image, bboxes, stage, debug_dir, image_name):
    for bbox in bboxes:
        label, x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * image.shape[1])
        y1 = int((y_center - height / 2) * image.shape[0])
        x2 = int((x_center + width / 2) * image.shape[1])
        y2 = int((y_center + height / 2) * image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    output_path = os.path.join(debug_dir, f"{os.path.splitext(image_name)[0]}_{stage}.jpg")
    cv2.imwrite(output_path, image)

def prepare_dataset(image_dir, label_dir, output_dir, debug_dir):
    img_output_dir = os.path.join(output_dir, 'images')
    lbl_output_dir = os.path.join(output_dir, 'labels')

    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))
    for img_path in image_files:
        img = cv2.imread(img_path)
        original_height, original_width = img.shape[:2]
        image_name = os.path.basename(img_path)

        lbl_path = os.path.splitext(image_name)[0] + '.txt'
        src_lbl_path = os.path.join(label_dir, lbl_path)

        if not os.path.exists(src_lbl_path):
            continue

        with open(src_lbl_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f]
        if len(bboxes) != 10:
            continue

        # 원본 이미지를 640으로 리사이즈하여 저장
        img_resized = cv2.resize(img, (640, 640))
        img_output_path = os.path.join(img_output_dir, image_name)
        cv2.imwrite(img_output_path, img_resized)

        dst_lbl_path = os.path.join(lbl_output_dir, os.path.basename(lbl_path))

        adjusted_bboxes = adjust_bboxes_for_resized_image(bboxes, original_width, original_height, 640, 640)

        save_debug_image(img_resized.copy(), adjusted_bboxes, "resized", debug_dir, image_name)

        with open(dst_lbl_path, 'w') as f:
            for bbox in adjusted_bboxes:
                label, x_center, y_center, width, height = bbox
                f.write(f"{label} {x_center} {y_center} {width} {height}\n")

        # 스케일된 이미지를 저장
        for i, scaled_img in enumerate(image_pyramid(img)):
            img_resized = cv2.resize(scaled_img, (640, 640))
            scale_factor_x = original_width / scaled_img.shape[1]
            scale_factor_y = original_height / scaled_img.shape[0]

            img_output_path = os.path.join(img_output_dir, f"{os.path.splitext(image_name)[0]}_scale_{i}.jpg")
            cv2.imwrite(img_output_path, img_resized)

            scaled_lbl_path = os.path.join(lbl_output_dir, f"{os.path.splitext(lbl_path)[0]}_scale_{i}.txt")
            adjusted_bboxes = adjust_bboxes_for_resized_image(bboxes, original_width, original_height, scaled_img.shape[1], scaled_img.shape[0])

            save_debug_image(img_resized.copy(), adjusted_bboxes, f"scaled_{i}", debug_dir, image_name)

            with open(scaled_lbl_path, 'w') as f:
                for bbox in adjusted_bboxes:
                    label, x_center, y_center, width, height = bbox
                    f.write(f"{label} {x_center} {y_center} {width} {height}\n")

def train_with_early_stopping(model_path, data_yaml_path, epochs=100, patience=5):
    model = YOLO(model_path)
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        # 모델을 한 에포크 동안 학습시키고, 학습 결과를 저장합니다.
        results = model.train(data=data_yaml_path, epochs=1, batch=4, imgsz=640)

        # 학습 결과에서 손실 값을 추출합니다.
        current_loss = results.metrics.box_loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            epochs_no_improve = 0
            model.save(f"{model_path}_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = f"{script_dir}/images_01"
    label_file = f"{script_dir}/labels.json"
    output_dir = f"{script_dir}/yolo_dataset_new"
    debug_dir = f"{script_dir}/debug"
    
    label_dir = os.path.join(output_dir, 'labels')
    img_width, img_height = 1920, 1080

    convert_to_yolo_format(label_file, label_dir, img_width, img_height)
    prepare_dataset(image_dir, label_dir, output_dir, debug_dir)

    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    data_yaml_content = f"""
    path: {output_dir}
    train: images
    val: images
    nc: 10
    names: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    """
    
    with open(data_yaml_path, 'w') as f:
        f.write(data_yaml_content)

    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
    
    

    # 추가 데이터로 보강 학습
    # model_path = 'yolov8s.pt'
    # train_with_early_stopping(model_path, data_yaml_path, epochs=100, patience=5)

    model = YOLO('yolov8s.pt')
    # model.load(f"{script_dir}/runs/detect/train5/weights/best.pt")
    model.train(data=f"{output_dir}/data.yaml", epochs=50, batch=2, imgsz=640)
