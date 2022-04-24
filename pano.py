import cv2
import numpy as np
import sift

# 상수 변수와 데이터 클래스
SIZE = 5
FILTER = np.ones((SIZE, SIZE)) / (SIZE * SIZE)
BLACK = [10, 10, 10]


class ImagePosition:
    def __init__(self, homo, dx):
        self.h = homo
        self.dx = dx


class PANO:
    def __int__(self):
        self.l_img = None
        self.r_img = None
        self.pano = None

    def __init__(self, left_image, right_image):
        self.l_img = left_image
        self.r_img = right_image


def compare_color(color1, color2):
    """
    현재 픽셀 내 BGR 값을 받아 온 뒤 비교 대상군 보다 값이 작은지 파악 후 AND 연산을 실행 하는 함수
    :param color1: 픽셀 BGR 값
    :param color2: 비교 대상군 BGR (현재 스크립트에서는 BLACK 에 해당함)
    :return: 셋다 참일 경우에만 TRUE를 반환.
    """
    arr = color1 <= color2
    result = 1
    for i in arr:
        result *= i
    return result


def calc_overlap(img1, img2):
    """
    겹치는 부분을 계산하는 함수. 겹치는 부분은 픽셀 연산으로 계산한다.

    :param img1: 왼쪽 이미지
    :param img2: 오른쪽 이미지
    :return: 크롭 되지 않는 겹치는 부분 마스크, (겹치는 부분의 너비 부분 픽셀의 최소값, 최대값)
   """
    # threshold 를 통한 마스크 연산을 통해, 겹치는 부분을 구한다. (가로 길이가 넓은 영상)
    _, data_map = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    temp = cv2.copyTo(img2, data_map)
    _, data_map2 = cv2.threshold(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    dt = np.asarray(data_map2)
    min_array = -1
    max_array = -1
    for h in dt:
        temp_min = -1
        temp_max = -1
        flag = 0
        for i, w in enumerate(h):
            if flag == 0:
                if w > 0:
                    flag = 1
                    temp_min = i
            elif flag == 1:
                if w == 0:
                    flag = 0
                    temp_max = i
        if temp_min >= 0 or temp_max >= 0:
            if min_array < 0 or min_array > temp_min:
                min_array = temp_min
            if max_array < 0 or max_array < temp_max:
                max_array = temp_max
    data_map2 = cv2.cvtColor(data_map2, cv2.COLOR_GRAY2BGR)
    return data_map2, (max_array, min_array)


# panorama 메인 클래스
class PANORAMA:
    """
    :내부 변수:
    homograph = 사진 변환에 사용할 초기 호모그래프 행렬
    status = KnnMatcher를 사용하여 호모그래프를 구할 때 함께 나오는 상태. (사용되지 않음)
    ip = ImagePosition 클래스 자료형의 객체 (호모그래피와 dx를 담고있다.)
    overlap_w = 겹치는 부분의 가로 픽셀값 (왼쪽끝과 오른쪽 끝)
    """
    __homograph = None
    __status = None
    __ip = None
    __overlap_w = None

    # panorama class
    def __init__(self, image1, image2):
        """
        1. 이미지를 받아오면 각각 sift 객체를 생성.
        2. 생성된 sift 객체에서 SIFT 연산을 통해 특징점을 찾아냄
        3. 찾아낸 특징점을 KnnMatcher를 사용하여 호모그래프를 찾아냄
        4. 영상의 위치를 계산하여 호모그래피를 변경하고, dx를 구하여 ip에 저장함.
        5. 호모그래프를 사용하여 이미지를 변형한 뒤, PANO 클래스에 left, right 이미지를 각각 저장.
            (원본 이미지를 저장하는 것이 아닌 변형된 이미지를 저장함)
        6. 저장된 데이터들을 바탕으로 파노라마 이미지를 제작함.
        """
        self.sift1 = sift.SIFT(image1)
        self.sift2 = sift.SIFT(image2)
        k = sift.KnnMatcher(self.sift1, self.sift2)
        self.__homograph, self.__status = k.H, k.status
        self.__img_position()
        self.__make_trans_img()
        self.__construct_panorama()

    def show(self):
        """
        저장된 파노라마 이미지를 보여주는 함수
        """
        if self.panorama is not None:
            cv2.imshow("result", self.panorama.pano)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def __img_position(self):
        homo = np.ones((3, 1))
        homo[0, 0] = self.sift2.img.shape[1]
        homo[1, 0] = self.sift2.img.shape[0]
        """
        img2 의 width 와 height 
                |  w  |
        homo =  |  h  |
                |  1  |
        """
        new_homo = np.dot(self.__homograph, homo)
        """
                    |               |     |  w  |
        new_homo =  |   homograph   |  x  |  h  |
                    |    (3 x 3)    |     |  1  |
        """
        for i in range(new_homo.shape[0]):
            new_homo[i, 0] = new_homo[i, 0] / new_homo[2, 0]
        """
        |  NH0  |    |  NH0 / NH2  |
        |  NH1  | =  |  NH1 / NH2  |
        |  NH2  |    |      1      |
        변형 후 img2 의 width (NH0 / NH2) 와 height (NH1 / NH2)
        """
        dx = int(new_homo[0, 0] - homo[0, 0])
        """
        Panorama 로 연결할 위치를 파악
        dx가 음수 -> img2가 img1보다 왼쪽에 위치　      ( 결과 이미지 = img2 ~ img1 )
        dx가 양수 -> img2가 img1보다 오른쪽에 위치      ( 결과 이미지 = img1 ~ img2 )
        """
        if dx < 0:
            dxt = np.absolute(dx)
            new_h = np.array([[1, 0, dxt], [0, 1, 0], [0, 0, 1]], np.float32)
            homograph = np.dot(new_h, self.__homograph)
        else:
            homograph = self.__homograph
        """
        dx가 음수 라면 ( 왼쪽에 위치 한다면) homograph 를 아래와 같이 변환함.
                      |  1   0  dxt |     |               |     |  h00 + dxt x h20  h01 + dxt x h21  h02 + dxt x h22  |
        homograph  =  |  0   1   0  |  x  |   homograph   |  =  |        h10              h11              h12        |
                      |  0   0   1  |     |    (3 x 3)    |     |        h20              h21              h22        |
        """
        self.__ip = ImagePosition(homograph, dx)

    def __make_trans_img(self):
        dxt = np.absolute(self.__ip.dx)
        y = self.sift1.img.shape[0]
        x = self.sift1.img.shape[1] + dxt
        non_transformed_image = np.zeros((y, x, 3), np.uint8)
        if self.__ip.dx < 0:
            transformed_image = cv2.warpPerspective(self.sift2.img, self.__ip.h, (x, y), flags=cv2.INTER_LINEAR)
            non_transformed_image[:, dxt:] = self.sift1.img  # 왼쪽 합성
            self.panorama = PANO(left_image=transformed_image, right_image=non_transformed_image)
        else:
            transformed_image = cv2.warpPerspective(self.sift2.img, self.__ip.h,
                                                    (self.sift2.img.shape[1] + dxt, self.sift2.img.shape[0]),
                                                    flags=cv2.INTER_LINEAR)
            non_transformed_image[:, :self.sift1.img.shape[1]] = self.sift1.img[:, :self.sift1.img.shape[1]]  # 오른쪽 합성
            self.panorama = PANO(left_image=non_transformed_image, right_image=transformed_image)
        _, self.__overlap_w = calc_overlap(transformed_image, non_transformed_image)
        self.panorama.pano = np.zeros((y, x, 3), np.uint8)

    def __construct_panorama(self):
        self.panorama.pano[:, :self.__overlap_w[1]] = self.panorama.l_img[:, :self.__overlap_w[1]]
        self.panorama.pano[:, self.__overlap_w[0]:] = self.panorama.r_img[:, self.__overlap_w[0]:]
        """
        오른쪽 이미지에 가해지는 weight와 왼쪽 이미지에 가해지는 weight를 만든다.
        오른쪽 weight는 0부터 1까지 overlpa_w 픽셀 수만큼의 크기인 등차수열이며
        왼쪽 weight는 오른쪽 weight를 뒤집은 수열이다.
        """
        w_right = np.arange(0, 1, 1 / (self.__overlap_w[0] - self.__overlap_w[1]))
        w_left = [1. - x for x in w_right]
        left_mid = self.panorama.l_img[:, self.__overlap_w[1]:self.__overlap_w[0]]
        right_mid = self.panorama.r_img[:, self.__overlap_w[1]:self.__overlap_w[0]]
        mid = np.zeros(left_mid.shape, np.uint8)
        """
        각각의 픽셀에 weight를 곱하는 픽셀연산을 시행한다.
        compare_color함수를 통하여 만약 어느 한쪽이 BLACK 보다 작은 값의 경우 
        다른쪽의 픽셀값을 그대로 가져온다.
        (예를 들어, 오른쪽 이미지의 특정 부분이 검은색이면 왼쪽 이미지의 특정 부분 픽셀값을 그대로 가져온다.)
        그 외의 경우엔, 위에서 구했던 weight를 사용한 연산을 시행한다.
        """
        for h, v in enumerate(mid):
            for w, _ in enumerate(v):
                if compare_color(left_mid[h][w], BLACK):
                    mid[h][w] = right_mid[h][w]
                elif compare_color(right_mid[h][w], BLACK):
                    mid[h][w] = left_mid[h][w]
                else:
                    mid[h][w] = left_mid[h][w] * w_left[w] + right_mid[h][w] * w_right[w]
        self.panorama.pano[:, self.__overlap_w[1]:self.__overlap_w[0]] = mid
