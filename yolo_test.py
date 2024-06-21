import os
import cv2
from ultralytics import YOLO
import glob
import json
import numpy as np

def load_and_predict(image_dir, model_path, label_file):
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 이미지 디렉토리에서 모든 이미지 파일 읽기
    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))
    index = 0

    while index < len(image_files):
        image_file = image_files[index]

        # 이미지 불러오기
        image = cv2.imread(image_file)  # RGB 이미지로 읽기
        original_image = image.copy()

        # 디텍션 수행
        results = model.predict(image, imgsz=640)

        # 결과 이미지 그리기
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                label = f"{int(box.cls[0].cpu().item())} {box.conf[0].cpu().item():.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # label.json 파일 다시 읽기
        with open(label_file, 'r') as f:
            label_data = json.load(f)

        # 라벨 파일에서 해당 이미지의 라벨 가져오기
        image_name = os.path.basename(image_file)
        image_labels = next((item for item in label_data["images"] if item["file"] == image_name), None)

        if image_labels:
            for ann in image_labels["annotations"]:
                label = ann["label"]
                bbox = ann["bbox"]
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 이미지 이름 표시
        cv2.putText(image, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 결과 이미지 표시
        cv2.imshow('Detection Result', image)

        # Manual Mode for labeling
        manual_squares = []
        while True:
            key = cv2.waitKey(0)

            if key == ord('n'):
                index += 1
                break
            elif key == ord('b'):
                index = max(0, index - 1)
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):
                save_label_data(image_name, manual_squares, label_file)
                print("Labels saved.")
                break
            elif key == ord('m'):
                manual_squares = manual_label(original_image)

def save_label_data(image_file, manual_squares, label_file):
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            label_data = json.load(f)
    else:
        label_data = {"images": []}

    # 이미지 이름으로 기존 데이터 찾기
    image_name = os.path.basename(image_file)
    existing_image_data = next((item for item in label_data["images"] if item["file"] == image_name), None)

    if existing_image_data is not None:
        # 기존 데이터에 새 데이터를 추가
        for sq, recognized_digit in manual_squares:
            x, y, w, h = cv2.boundingRect(np.array(sq))
            existing_image_data["annotations"].append({"label": str(recognized_digit), "bbox": [x, y, x + w, y + h]})
    else:
        # 새로운 데이터를 추가
        image_data = {
            "file": image_name,
            "annotations": []
        }
        for sq, recognized_digit in manual_squares:
            x, y, w, h = cv2.boundingRect(np.array(sq))
            image_data["annotations"].append({"label": str(recognized_digit), "bbox": [x, y, x + w, y + h]})
        label_data["images"].append(image_data)

    with open(label_file, 'w') as f:
        json.dump(label_data, f, indent=4)

def manual_label(image):
    manual_squares = []

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, ex, ey, current_rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                ex, ey = x, y
                current_rectangle = [(ix, iy), (ex, ey)]
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            ex, ey = x, y
            current_rectangle = [(ix, iy), (ex, ey)]
            label = input(f"Enter label for the rectangle at ({ix},{iy}) to ({ex},{ey}): ")
            if label.isdigit():
                manual_squares.append((current_rectangle, int(label)))

    drawing = False
    ix, iy = -1, -1
    ex, ey = -1, -1
    current_rectangle = []

    cv2.namedWindow('Manual Labeling')
    cv2.setMouseCallback('Manual Labeling', draw_rectangle)

    while True:
        temp_image = image.copy()
        if current_rectangle:
            cv2.rectangle(temp_image, current_rectangle[0], current_rectangle[1], (0, 255, 0), 2)
        
        for sq, digit in manual_squares:
            x, y, w, h = cv2.boundingRect(np.array(sq))
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(temp_image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Manual Labeling', temp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save and exit
            cv2.destroyWindow('Manual Labeling')
            return manual_squares
        elif key == ord('q'):  # Quit without saving
            cv2.destroyWindow('Manual Labeling')
            return []

    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = f"{script_dir}/images_01"
    model_path = f"{script_dir}/runs/detect/train13/weights/best.pt"
    label_file = f"{script_dir}/labels.json"

    load_and_predict(image_dir, model_path, label_file)
