import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numpy.linalg import svd
import copy
import cv2
from cv2 import dct, idct
from pywt import dwt2, idwt2

class WaterMarkCore:
    def __init__(self, password_img=1, mode='common', processes=None):
        self.block_shape = np.array([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = 36, 20  # d1/d2가 클수록 견고성이 강하지만, 출력 이미지의 왜곡이 큽니다.

        # 초기화 데이터
        self.img, self.img_YUV = None, None  # self.img는 원본 이미지, self.img_YUV는 픽셀을 흰색으로 채운 짝수화한 이미지입니다.
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # 각 채널의 dct 결과
        self.ca_block = [np.array([])] * 3  # 각 채널이 4차원 배열을 저장하며, 4차원 블록의 결과를 나타냅니다.
        self.ca_part = [np.array([])] * 3  # 4차원 블록 후, 일부가 정수가 아니어서 누락된 부분을 나타냅니다.
        self.wm_size, self.block_num = 0, 0  # 워터마크 길이와 원본 이미지에 삽입할 수 있는 정보의 개수
        self.fast_mode = False
        self.alpha = None  # 투명 이미지 처리를 위해 사용됩니다.

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '최대 {}kb 정보를 삽입할 수 있으며, 워터마크 정보 {}kb를 초과하여 삽입할 수 없습니다.'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape는 정수로 변환된 ca의 2차원 크기입니다. 삽입 시 오른쪽과 아래쪽의 맞지 않는 부분을 무시합니다.
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img):
        # 투명 이미지 처리
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        # 이미지를 읽고 -> YUV로 변환 -> 흰색 경계 추가하여 픽셀을 짝수화 -> 4차원 블록으로 분할
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        # 짝수가 아니면 흰색 경계를 추가합니다. Y(밝기), UV(색상)
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 4차원으로 변환
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def block_add_wm(self, arg):
        if self.fast_mode:
            return self.block_add_wm_fast(arg)
        else:
            return self.block_add_wm_slow(arg)

    def block_add_wm_slow(self, arg):
        block, i = arg
        # dct->svd->워터마크 삽입->역svd->역dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = dct(block)

        u, s, v = svd(block_dct)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2

        block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
        return idct(block_dct_flatten.reshape(self.block_shape))

    def block_add_wm_fast(self, arg):
        # dct->svd->워터마크 삽입->역svd->역dct
        block, i = arg
        wm_1 = self.wm_bit[i % self.wm_size]

        u, s, v = svd(dct(block))
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1

        return idct(np.dot(u, np.dot(np.diag(s), v)))

    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3

        for channel in range(3):
            tmp = [(self.ca_block[channel][self.block_index[i]], i) for i in range(self.block_num)]

            for i, result in enumerate(map(self.block_add_wm, tmp)):
                self.ca_block[channel][self.block_index[i]] = result

            # 4차원 블록을 2차원으로 변환
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # 4차원 블록 시 오른쪽과 아래쪽이 맞지 않는 길이는 유지하고, 나머지 부분은 워터마크가 삽입된 주파수 영역 데이터로 바꿉니다.
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # 역변환하여 되돌립니다.
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # 3채널 합치기
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 이전에 짝수가 아닌 경우 흰색 경계를 추가했으므로, 여기서 제거합니다.
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def block_get_wm(self, args):
        if self.fast_mode:
            return self.block_get_wm_fast(args)
        else:
            return self.block_get_wm_slow(args)

    def block_get_wm_slow(self, args):
        block = args
        # dct->svd->워터마크 추출
        block_dct = dct(block)

        u, s, v = svd(block_dct)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def block_get_wm_fast(self, args):
        block = args
        # dct->svd->워터마크 추출
        u, s, v = svd(dct(block))
        wm = (s[0] % self.d1 > self.d1 / 2) * 1

        return wm

    def extract_raw(self, img):
        # 각 블록에서 1 bit 정보 추출
        self.read_img_arr(img=img)
        self.init_block_index()

        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3개 채널, 각 블록에서 추출한 워터마크를 저장

        for channel in range(3):
            wm_block_bit[channel, :] = [self.block_get_wm(self.ca_block[channel][self.block_index[i]])
                                        for i in range(self.block_num)]
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # 순환 삽입 및 3개 채널 평균 계산
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()

        # 각 블록에 삽입된 비트를 추출
        wm_block_bit = self.extract_raw(img=img)
        # 평균 계산
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        wm_avg = self.extract(img=img, wm_shape=wm_shape)

        return one_dim_kmeans(wm_avg)

def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  # 1. 초기 중심점
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  # 2. 모든 점과 이 k개의 점 사이의 거리를 확인하고, 각 점을 가장 가까운 중심에 할당
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]  # 3. 중심점 다시 찾기
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:  # 4. 종료 조건
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01