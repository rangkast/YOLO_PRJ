import os
import cv2
from ultralytics import YOLO
import glob

def load_and_predict(image_dir, model_path):
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 이미지 디렉토리에서 모든 이미지 파일 읽기
    image_files = sorted(glob.glob(f"{image_dir}/*.jpg"))

    for image_file in image_files:
        # 이미지 불러오기
        image = cv2.imread(image_file)  # RGB 이미지로 읽기

        # 디텍션 수행
        results = model.predict(image, imgsz=320)

        # 결과 이미지 그리기
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                label = f"{int(box.cls[0].cpu().item())} {box.conf[0].cpu().item():.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 결과 이미지 표시
        cv2.imshow('Detection Result', image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 'q' 키를 누르면 종료
        if key == ord('q'):
            break

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = f"{script_dir}/images"
    model_path = f"{script_dir}/runs/detect/train/weights/best.pt"

    load_and_predict(image_dir, model_path)
