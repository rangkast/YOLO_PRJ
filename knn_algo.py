
import cv2
import numpy as np
import os
import pickle

class KNNDetector:
    def __init__(self, knn_model_path=None, knn_image_size=(128, 128), k=5):
        self.knn_image_size = knn_image_size
        self.k = k
        self.knn_model = None
        if knn_model_path:
            self.load_knn_model(knn_model_path)

    def augment_image(self, img):
        augmented_images = [img]

        # 다양한 변형 적용
        rows, cols = img.shape

        # 회전
        for angle in range(-10, 11, 5):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            dst = cv2.warpAffine(img, M, (cols, rows))
            augmented_images.append(dst)

        # 이동
        for tx in range(-3, 4, 3):
            for ty in range(-3, 4, 3):
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                dst = cv2.warpAffine(img, M, (cols, rows))
                augmented_images.append(dst)

        return augmented_images

    def load_knn_images(self, image_dir, num_signs):
        symbols = []
        knn_images = []
        knn_labels = []
        for i in range(num_signs):
            filename = f"{image_dir}/T{i}.jpg"
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            symbols.append({"img": bin_img, "name": str(i)})

            # 증강된 이미지를 사용하여 학습 데이터 생성
            augmented_images = self.augment_image(bin_img)
            for aug_img in augmented_images:
                knn_img = cv2.resize(aug_img, self.knn_image_size)
                knn_images.append(knn_img.reshape(-1, np.prod(self.knn_image_size)).astype(np.float32))
                knn_labels.append(int(i))
        
        knn_images = np.array(knn_images).reshape(-1, np.prod(self.knn_image_size)).astype(np.float32)
        knn_labels = np.array(knn_labels).astype(np.int32)
        
        return symbols, knn_images, knn_labels

    def train_knn(self, knn_images, knn_labels):
        knn = cv2.ml.KNearest_create()
        knn.setDefaultK(self.k)
        knn.train(knn_images, cv2.ml.ROW_SAMPLE, knn_labels)
        self.knn_model = knn

    def unsharp_mask(self, image, sigma=1.0, strength=1.0):
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened

    def pyramid_image_up(self, image, scales):
        images = [image]
        for i in range(1, scales):
            scaled = cv2.pyrUp(images[-1])
            images.append(scaled)
        return images

    def recognize_digits(self, image, draw_image, squares):
        recognized_digits = []
        scales = 3  # 피라미드 업 스케일 수

        for sq in squares:
            x, y, w, h = cv2.boundingRect(sq)
            
            # 박스 크기를 조금 작게 만들어서 숫자에 더 가깝게
            margin_x, margin_y = int(w * 0.1), int(h * 0.1)
            x = max(x + margin_x, 0)
            y = max(y + margin_y, 0)
            w = max(w - 2 * margin_x, 1)
            h = max(h - 2 * margin_y, 1)

            roi = image[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = self.unsharp_mask(roi_gray, sigma=1.5, strength=1.5)  # 이미지 선명화
            roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            roi_pyramid = self.pyramid_image_up(roi_gray, scales)
            
            recognized_digit = None
            best_dist = float('inf')

            for roi_scaled in roi_pyramid:
                roi_scaled = cv2.resize(roi_scaled, self.knn_image_size)
                _, roi_thresh = cv2.threshold(roi_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                roi_flat = roi_thresh.reshape(-1, np.prod(self.knn_image_size)).astype(np.float32)

                ret, result, neighbours, dist = self.knn_model.findNearest(roi_flat, k=5)
                knn_digit = int(result[0][0])
                avg_dist = np.mean(dist)
                
                if avg_dist < best_dist:
                    best_dist = avg_dist
                    recognized_digit = knn_digit

            recognized_digits.append((sq, recognized_digit))

            # 사각형 그리기 및 레이블 추가
            # cv2.polylines(draw_image, [sq], True, (0, 255, 0), 2)
            label = f"{recognized_digit}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # cv2.rectangle(draw_image, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 255, 0), cv2.FILLED)
            # cv2.putText(draw_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return recognized_digits

    def save_knn_model(self, knn_images, knn_labels, filename='knn_train_data.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump((knn_images, knn_labels), f)

    def load_knn_model(self, filename='knn_train_data.pkl'):
        with open(filename, 'rb') as f:
            knn_images, knn_labels = pickle.load(f)
        self.train_knn(knn_images, knn_labels)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ref_image_dir = f"{script_dir}/../../../Targets/"
    knn_detector = KNNDetector(knn_image_size=(128, 128), k=5)
    symbols, knn_images, knn_labels = knn_detector.load_knn_images(ref_image_dir, num_signs=10)  # 
    knn_detector.train_knn(knn_images, knn_labels)
    knn_detector.save_knn_model(knn_images, knn_labels)