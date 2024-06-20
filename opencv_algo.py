import cv2
import numpy as np



def draw_squares(img, squares):
    for square, digit in squares:
        cv2.polylines(img, [square], True, (0, 0, 255), 3)
        # x, y = np.mean(square, axis=0).astype(int)
        # cv2.putText(img, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def match_digits(image, draw_image, squares, ref_images):
    matched_squares = []

    for sq in squares:
        x, y, w, h = cv2.boundingRect(sq)
        roi = image[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's Binarization 적용
        roi = cv2.resize(roi, (ref_images[0]['img'].shape[1], ref_images[0]['img'].shape[0]))

        min_diff = float('inf')
        recognized_digit = None
        best_match_img = None

        for ref_image in ref_images:
            diff = cv2.absdiff(roi, ref_image['img'])
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count < min_diff:
                min_diff = non_zero_count
                recognized_digit = ref_image['name']
                best_match_img = ref_image['img']

        # # 매칭된 순간에만 비교 이미지를 표시
        # if best_match_img is not None:
        #     cv2.imshow(f"ROI vs {recognized_digit}", np.hstack((roi, best_match_img)))
        #     cv2.waitKey(0)  # 키 입력을 대기

        # print(f"Recognized digit: {recognized_digit} with min_diff: {min_diff}")
        
        # 사각형 그리기
        # cv2.polylines(draw_image, [sq], True, (0, 0, 255), 3)
        
        # 숫자 추가
        if recognized_digit is not None:
            label = f"Digit: {recognized_digit}"
            # label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # cv2.rectangle(draw_image, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 0, 255), cv2.FILLED)
            # cv2.putText(draw_image, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        matched_squares.append((sq, recognized_digit))  # 사각형과 인식된 숫자 저장

    # print("\n")
    # draw_squares(draw_image, matched_squares)
    return matched_squares


def angle(pt1, pt2, pt0):
    dx1, dy1 = pt1[0] - pt0[0], pt1[1] - pt0[1]
    dx2, dy2 = pt2[0] - pt0[0], pt2[1] - pt0[1]
    dot_product = dx1 * dx2 + dy1 * dy2
    norm1 = dx1**2 + dy1**2
    norm2 = dx2**2 + dy2**2
    norm_product = np.sqrt(norm1 * norm2 + 1e-10)  # Avoid division by zero
    return dot_product / norm_product

def find_squares(image):
    squares = []

    # down-scale and upscale the image to filter out the noise
    pyr = cv2.pyrDown(image)
    timg = cv2.pyrUp(pyr)

    channels = cv2.split(timg)
    for c in range(3):  # 각 색상 채널에 대해 사각형을 검출
        gray0 = channels[c]

        for l in range(2):  # 여러 임계값 레벨을 시도
            if l == 0:
                gray = cv2.Canny(gray0, 0, 50, apertureSize=5)
                gray = cv2.dilate(gray, None)
            else:
                retval, gray = cv2.threshold(gray0, (l + 1) * 255 / 2, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_length = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * contour_length, True)
                area = cv2.contourArea(contour)

                # 면적 조건 추가 (1000 < area < 11000)
                if len(approx) == 4 and cv2.isContourConvex(approx) and 300 < area < 11000:
                    max_cosine = 0
                    for j in range(2, 5):
                        cosine = abs(angle(approx[j % 4][0], approx[j - 2][0], approx[j - 1][0]))
                        max_cosine = max(max_cosine, cosine)
                    if max_cosine < 0.3:
                        squares.append(approx.reshape(-1, 2))

    # 중복된 사각형 제거
    centers = [np.mean(sq, axis=0) for sq in squares]
    unique_squares = []
    added_centers = []

    for i, sq in enumerate(squares):
        center = centers[i]
        if not any(np.linalg.norm(center - ac) < 10 for ac in added_centers):
            unique_squares.append(sq)
            added_centers.append(center)

    # 정사각형 조건 강화
    final_squares = []
    for sq in unique_squares:
        side_lengths = [np.linalg.norm(sq[i] - sq[(i + 1) % 4]) for i in range(4)]
        if max(side_lengths) / min(side_lengths) < 1.3:  # 변의 길이가 비슷한 사각형만
            final_squares.append(sq)

    return final_squares


def draw_squares(img, squares):
    for square, digit in squares:
        cv2.polylines(img, [square], True, (0, 0, 255), 3)
        # x, y = np.mean(square, axis=0).astype(int)
        # cv2.putText(img, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def match_digits(image, draw_image, squares, ref_images):
    matched_squares = []

    for sq in squares:
        x, y, w, h = cv2.boundingRect(sq)
        roi = image[y:y+h, x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's Binarization 적용
        roi = cv2.resize(roi, (ref_images[0]['img'].shape[1], ref_images[0]['img'].shape[0]))

        min_diff = float('inf')
        recognized_digit = None
        best_match_img = None

        for ref_image in ref_images:
            diff = cv2.absdiff(roi, ref_image['img'])
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count < min_diff:
                min_diff = non_zero_count
                recognized_digit = ref_image['name']
                best_match_img = ref_image['img']

        # # 매칭된 순간에만 비교 이미지를 표시
        # if best_match_img is not None:
        #     cv2.imshow(f"ROI vs {recognized_digit}", np.hstack((roi, best_match_img)))
        #     cv2.waitKey(0)  # 키 입력을 대기

        # print(f"Recognized digit: {recognized_digit} with min_diff: {min_diff}")
        
        # 사각형 그리기
        # cv2.polylines(draw_image, [sq], True, (0, 0, 255), 3)
        
        # 숫자 추가
        if recognized_digit is not None:
            label = f"Digit: {recognized_digit}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # cv2.rectangle(draw_image, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 0, 255), cv2.FILLED)
            # cv2.putText(draw_image, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        matched_squares.append((sq, recognized_digit))  # 사각형과 인식된 숫자 저장

    # print("\n")
    # draw_squares(draw_image, matched_squares)
    return matched_squares




# from skimage.metrics import structural_similarity as ssim

# def unsharp_mask(image, sigma=1.0, strength=1.0):
#     blurred = cv2.GaussianBlur(image, (0, 0), sigma)
#     sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
#     return sharpened

# def pyramid_image_up(image, scales):
#     # 이미지 피라미드 업 생성
#     images = [image]
#     for i in range(1, scales):
#         scaled = cv2.pyrUp(images[-1])
#         images.append(scaled)
#     return images

# def resize_image_to_match(image, reference):
#     # 두 이미지를 동일한 크기로 리사이즈
#     h, w = reference.shape
#     return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

# def match_digits(image, draw_image, squares, ref_images, scales=3):
#     matched_squares = []
#     ssim_threshold = 0.5  # SSIM 임계값 설정

#     # 참조 이미지를 피라미드 업 형태로 생성
#     ref_pyramids = {ref_image['name']: pyramid_image_up(ref_image['img'], scales) for ref_image in ref_images}

#     for i, sq in enumerate(squares):
#         x, y, w, h = cv2.boundingRect(sq)
        
#         margin_x, margin_y = int(w * 0.02), int(h * 0.02)
#         x = max(x + margin_x, 0)
#         y = max(y + margin_y, 0)
#         w = max(w - 2 * margin_x, 1)
#         h = max(h - 2 * margin_y, 1)

#         roi = image[y:y+h, x:x+w]
#         roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         # 언샤프 마스크 필터 적용하여 선명화
#         roi = unsharp_mask(roi)

#         recognized_digit = None
#         best_match_img = None
#         max_ssim = -1

#         for ref_image in ref_images:
#             ref_pyramid = ref_pyramids[ref_image['name']]
#             for level in range(scales):
#                 roi_scaled = cv2.resize(roi, (ref_pyramid[level].shape[1], ref_pyramid[level].shape[0]))
#                 score, _ = ssim(roi_scaled, ref_pyramid[level], full=True)
#                 print(f"Comparing ROI to {ref_image['name']} at scale {level}, SSIM: {score}")
#                 if score > max_ssim:
#                     max_ssim = score
#                     recognized_digit = ref_image['name']
#                     best_match_img = ref_pyramid[level]

#         if max_ssim < ssim_threshold:
#             recognized_digit = None
#             print(f"ROI {i}의 SSIM 값 {max_ssim}이(가) 임계값 {ssim_threshold} 이하이므로 인식되지 않음")

#         print(f"Best match for ROI {i}: {recognized_digit} with SSIM: {max_ssim}")
        
#         cv2.polylines(draw_image, [sq], True, (0, 0, 255), 3)
        
#         if recognized_digit is not None:
#             label = f"Digit: {recognized_digit}"
#             cv2.putText(draw_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         matched_squares.append((sq, recognized_digit))
        
#         # 디버깅을 위해 ROI와 매칭된 숫자 이미지를 디스플레이
#         if best_match_img is not None:
#             best_match_img_resized = resize_image_to_match(best_match_img, roi)
#             debug_image = np.hstack((roi, best_match_img_resized))
#             cv2.imshow(f"ROI_{i}_vs_{recognized_digit}", debug_image)
#             cv2.waitKey(0)  # 키 입력 대기

#     return matched_squares
