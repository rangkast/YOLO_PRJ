import os
import cv2
import glob
import numpy as np
import json
import time
from yolov8_algo import YOLO_Detector
from knn_algo import KNNDetector
from opencv_algo import match_digits

CAP_PROP_FRAME_WIDTH = 1920
CAP_PROP_FRAME_HEIGHT = 1080

script_dir = os.path.dirname(os.path.realpath(__file__))

class YOLOAlgorithm:
    def __init__(self, model_path):
        self.detector = YOLO_Detector(model_path)
        self.ready = False
        self.model_name = self.detector.model_name

    def initialize(self):
        self.ready = True

    def detect(self, image, draw_image):
        start_time = time.time()
        result, box_data = self.detector.detect(image, draw_image)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Processing time: {elapsed_time_ms:.2f} ms")
        return result, box_data, elapsed_time_ms

def save_label_data(image_file, matched_squares, label_file):
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            label_data = json.load(f)
    else:
        label_data = {"images": []}

    # 이미지 이름으로 기존 데이터 찾기
    image_name = os.path.basename(image_file)
    existing_image_data = next((item for item in label_data["images"] if item["file"] == image_name), None)

    if existing_image_data is not None:
        # 기존 데이터를 업데이트
        existing_image_data["annotations"] = []
        for sq, recognized_digit in matched_squares:
            x, y, w, h = cv2.boundingRect(np.array(sq))
            existing_image_data["annotations"].append({"label": str(recognized_digit), "bbox": [x, y, x + w, y + h]})
    else:
        # 새로운 데이터를 추가
        image_data = {
            "file": image_name,
            "annotations": []
        }
        for sq, recognized_digit in matched_squares:
            x, y, w, h = cv2.boundingRect(np.array(sq))
            image_data["annotations"].append({"label": str(recognized_digit), "bbox": [x, y, x + w, y + h]})
        label_data["images"].append(image_data)

    with open(label_file, 'w') as f:
        json.dump(label_data, f, indent=4)

def manual_label(image, matched_squares):
    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, ex, ey, current_rectangle, manual_squares, image_updated
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
            center_x = (ix + ex) // 2
            center_y = (iy + ey) // 2
            updated = False
            for i, (sq, digit) in enumerate(matched_squares):
                x, y, w, h = cv2.boundingRect(np.array(sq))
                if x <= center_x <= x + w and y <= center_y <= y + h:
                    matched_squares.pop(i)
                    updated = True
                    break
            if not updated:
                label = input(f"Enter label for the rectangle at ({ix},{iy}) to ({ex},{ey}): ")
                if label.isdigit():
                    manual_squares.append((current_rectangle, int(label)))
            image_updated = True

    drawing = False
    ix, iy = -1, -1
    ex, ey = -1, -1
    current_rectangle = []
    manual_squares = []
    image_updated = False

    cv2.namedWindow('Manual Labeling')
    cv2.setMouseCallback('Manual Labeling', draw_rectangle)

    while True:
        temp_image = image.copy()
        if current_rectangle:
            cv2.rectangle(temp_image, current_rectangle[0], current_rectangle[1], (0, 255, 0), 2)
        
        for sq, digit in matched_squares + manual_squares:
            x, y, w, h = cv2.boundingRect(np.array(sq))
            color = (0, 255, 0) if any((np.array_equal(sq, msq) and digit == mdigit) for msq, mdigit in manual_squares) else (255, 0, 0)
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(temp_image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Manual Labeling', temp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save and exit
            cv2.destroyWindow('Manual Labeling')
            return True, matched_squares + manual_squares
        elif key == ord('q'):  # Quit without saving
            cv2.destroyWindow('Manual Labeling')
            return False, manual_squares
        elif image_updated:
            image_updated = False

    cv2.destroyAllWindows()

def video_to_images(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        frame_count += 1

    cap.release()
    print("Finished saving frames.")

def draw_existing_labels(draw_frame, label_file, image_file):
    if not os.path.exists(label_file):
        return

    with open(label_file, 'r') as f:
        label_data = json.load(f)

    image_name = os.path.basename(image_file)
    existing_image_data = next((item for item in label_data["images"] if item["file"] == image_name), None)

    if existing_image_data is not None:
        for annotation in existing_image_data["annotations"]:
            x1, y1, x2, y2 = annotation["bbox"]
            label = annotation["label"]
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(draw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def view_images_with_algo(image_dir, algorithm, label_file, ref_images):
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    
    if not image_files:
        print("No images found in directory.")
        return

    algorithm.initialize()
    knn_detector = KNNDetector(knn_model_path=f"{script_dir}/models/knn_train_data.pkl", knn_image_size=(128, 128), k=5)
    for image_file in image_files:
        image = cv2.imread(image_file)
        print(f"Processing image: {os.path.basename(image_file)}")
        draw_frame = image.copy()
        _, box_data, exec_time = algorithm.detect(image, draw_frame)
        
        boxes = box_data[0]
        squares = [np.array([[x1, y1], [x2, y2]]) for (x1, y1, x2, y2) in boxes]

        matched_digit = match_digits(image, draw_frame, squares, ref_images)
        matched_knn = knn_detector.recognize_digits(image, draw_frame, squares)

        # 일치하는 항목 찾기
        matched_squares = []
        unmatched_squares = []

        for sq in squares:
            digit1 = next((digit for (sq1, digit) in matched_digit if np.array_equal(sq1, sq)), None)
            digit2 = next((digit for (sq2, digit) in matched_knn if np.array_equal(sq2, sq)), None)
            if digit1 == digit2 and digit1 is not None:
                matched_squares.append((sq, digit1))
            else:
                unmatched_squares.append((sq, digit1 or digit2))

        # 후보를 빨간색으로 표시
        for sq, digit in unmatched_squares:
            x, y, w, h = cv2.boundingRect(np.array(sq))
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(draw_frame, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 기존 라벨 표시
        draw_existing_labels(draw_frame, label_file, image_file)

        success, labeled_squares = manual_label(draw_frame, matched_squares)
        if success:
            save_label_data(image_file, labeled_squares, label_file)
            print("Labels saved.")

        print("Press 'n' for next frame, 'q' to quit")

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        elif key == ord('n'):
            continue

    cv2.destroyAllWindows()

def load_ref_images(image_dir, num_signs, scale=0.50):
    symbols = []
    for i in range(num_signs):
        filename = f"{image_dir}/T{i}.jpg"
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        symbols.append({"img": img, "name": str(i)})
    return symbols

def label_images(image_dir, label_file, ref_images, model_path):
    algorithm = YOLOAlgorithm(model_path)
    view_images_with_algo(image_dir, algorithm, label_file, ref_images)

if __name__ == "__main__":
    video_path = os.path.join(script_dir, '../../../Pytest/dataset/NewTrainingRaw/movies/dataset_1.mp4')
    output_dir = os.path.join(script_dir, 'images_01')
    label_file = os.path.join(script_dir, 'labels.json')

    ref_image_dir = f"{script_dir}/Targets/"
    num_signs = 10
    symbols = load_ref_images(ref_image_dir, num_signs)

    # Step 1: Convert video to images
    # video_to_images(video_path, output_dir)

    # Step 2: Label images using YOLO detection and match digits
    model_path = f"{script_dir}/runs/detect/train9/weights/best.pt"
    label_images(output_dir, label_file, symbols, model_path)
