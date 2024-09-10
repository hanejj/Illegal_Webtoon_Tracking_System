import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from bwm_core import WaterMarkCore

class WaterMark:
    def __init__(self, password_wm=1, block_shape=(4, 4), mode='common', processes=None):
        self.bwm_core = WaterMarkCore(mode=mode, processes=processes)
        self.password_wm = password_wm
        self.wm_bit = None
        self.wm_size = 0

    def read_img(self, filename=None, img=None):
        if img is None:
            img = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
            assert img is not None, f"이미지 파일 '{filename}'을(를) 읽지 못했습니다."

        self.bwm_core.read_img_arr(img=img)
        return img

    def read_wm(self, wm_content, mode='img'):
        assert mode in ('img', 'str', 'bit'), "mode는 ('img', 'str', 'bit') 중 하나여야 합니다."
        if mode == 'img':
            wm = cv2.imread(wm_content, flags=cv2.IMREAD_GRAYSCALE)
            assert wm is not None, f'파일 "{wm_content}"을(를) 읽지 못했습니다.'

            self.wm_bit = wm.flatten() > 128

        elif mode == 'str':
            byte = bin(int(wm_content.encode('utf-8').hex(), base=16))[2:]
            self.wm_bit = (np.array(list(byte)) == '1')
        else:
            self.wm_bit = np.array(wm_content)

        self.wm_size = self.wm_bit.size

        # 워터마크 암호화(섞기)
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)

        self.bwm_core.read_wm(self.wm_bit)

    def embed(self, filename=None, compression_ratio=None):
        embed_img = self.bwm_core.embed()
        if filename is not None:
            if compression_ratio is None:
                cv2.imwrite(filename=filename, img=embed_img)
            elif filename.endswith('.jpg'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_JPEG_QUALITY, compression_ratio])
            elif filename.endswith('.png'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_PNG_COMPRESSION, compression_ratio])
            else:
                cv2.imwrite(filename=filename, img=embed_img)
        return embed_img

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

    def extract(self, filename=None, embed_img=None, wm_shape=None, out_wm_name=None, mode='img'):
        assert wm_shape is not None, 'wm_shape가 필요합니다.'

        if filename is not None:
            embed_img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
            assert embed_img is not None, f"{filename}을(를) 읽지 못했습니다."

        self.wm_size = np.array(wm_shape).prod()

        if mode in ('str', 'bit'):
            wm_avg = self.bwm_core.extract_with_kmeans(img=embed_img, wm_shape=wm_shape)
        else:
            wm_avg = self.bwm_core.extract(img=embed_img, wm_shape=wm_shape)

        # 워터마크 복호화(원래 순서로 복원)
        wm = self.extract_decrypt(wm_avg=wm_avg)

        if mode == 'img':
            wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])
            cv2.imwrite(out_wm_name, wm)
        elif mode == 'str':
            byte = ''.join(str((i >= 0.5) * 1) for i in wm)
            wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode('utf-8', errors='replace')

        return wm

    def detect_pos(self, img_gray):
        """
        ORB를 사용하여 이미지에서 강력한 특징점을 탐지하고 반환합니다.
        :param img_gray: 그레이스케일 이미지
        :return: 특징점 좌표
        """
        orb = cv2.ORB_create()
        keypoints = orb.detect(img_gray, None)
        keypoints, _ = orb.compute(img_gray, keypoints)

        if not keypoints:
            raise ValueError("특징점을 찾을 수 없습니다.")

        return keypoints[0].pt

    def get_crop_coordinates(self, pos, img_shape, crop_size=(700, 700)):
        h, w = img_shape
        new_w, new_h = crop_size

        if pos[0] <= new_w // 2:
            x = 0
        elif pos[0] >= w - new_w // 2:
            x = w - new_w
        else:
            x = int(pos[0]) - new_w // 2

        if pos[1] <= new_h // 2:
            y = 0
        elif pos[1] >= h - new_h // 2:
            y = h - new_h
        else:
            y = int(pos[1]) - new_h // 2

        return x, y

    def crop_feature_based(self, img, crop_size=(700, 700)):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pos = self.detect_pos(img_gray)
        x, y = self.get_crop_coordinates(pos, img.shape[:2], crop_size)
        cropped_img = img[y:y + crop_size[1], x:x + crop_size[0]]
        return cropped_img, x, y

    def remove_noise(self, image_path, output_path):
        # 이미지 읽기
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 이진화 임계값 조정 (다양한 값 시도)
        thresholds = [40, 40, 60]
        best_result = None
        best_quality = 0

        for t in thresholds:
            _, binary_img = cv2.threshold(img, t, 255, cv2.THRESH_BINARY_INV)

            # 모폴로지 변환 (노이즈 제거)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
            cleaned_img = cv2.morphologyEx(cleaned_img, cv2.MORPH_OPEN, kernel)

            # 품질 평가 (여기서는 단순히 빈도 수를 사용)
            non_zero_count = np.count_nonzero(cleaned_img)
            if non_zero_count > best_quality:
                best_quality = non_zero_count
                best_result = cleaned_img

        # 결과 저장
        cv2.imwrite(output_path, best_result)

    def invert_image(self, image_path, output_path):
        # 이미지 읽기
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 반전
        inverted_img = cv2.bitwise_not(img)

        # 결과 저장
        cv2.imwrite(output_path, inverted_img)


if __name__ == "__main__":
    host_image_path = "host.png"
    watermark_image_path = "50B.png"
    output_image_path = "output_image.png"
    extracted_watermark_path = "extracted_watermark.png"
    cleaned_watermark_path = "cleaned_watermark.png"
    contrast_adjusted_path = "contrast_adjusted_watermark.png"
    inverted_watermark_path = "inverted_watermark.png"

    wm = WaterMark()

    # 호스트 이미지 읽기 및 크롭
    host_img = wm.read_img(filename=host_image_path)
    cropped_host_img, x, y = wm.crop_feature_based(host_img)
    cv2.imwrite("cropped_host.png", cropped_host_img)

    # 워터마크 삽입
    wm.read_wm(watermark_image_path, mode='img')
    embed_img = wm.embed()

    embed_cropped_img = embed_img[y:y + cropped_host_img.shape[0], x:x + cropped_host_img.shape[1]]

    if host_img.shape[2] == 4:
        alpha_channel = np.ones((cropped_host_img.shape[0], cropped_host_img.shape[1], 1), dtype=embed_cropped_img.dtype) * 255
        embed_cropped_img = np.concatenate((embed_cropped_img, alpha_channel), axis=2)

    host_img[y:y + cropped_host_img.shape[0], x:x + cropped_host_img.shape[1]] = embed_cropped_img

    cv2.imwrite(output_image_path, host_img)

    wm_shape = (100, 150)
    wm.extract(filename=output_image_path, wm_shape=wm_shape, out_wm_name=extracted_watermark_path)

    # 노이즈 제거 및 결과 저장
    wm.remove_noise(extracted_watermark_path, cleaned_watermark_path)

    # 이미지 반전 및 결과 저장
    wm.invert_image(cleaned_watermark_path, inverted_watermark_path)
