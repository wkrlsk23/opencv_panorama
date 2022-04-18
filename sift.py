import numpy as np
import cv2
import sys


class Imageposition:
    def __init__(self, homo, dx):
        self.h = homo
        self.dx = dx


class PANORAMA:
    __homograph = None
    __status = None
    __ip = None

    def __init__(self, image1, image2):
        self.panorama = None
        self.sift1 = SIFT(image1)
        self.sift2 = SIFT(image2)
        k = Knnmatcher(self.sift1, self.sift2)
        self.__homograph, self.__status = k.H, k.status
        self.__img_position()
        self.__make_trans_img()

    def show(self):
        if self.panorama is not None:
            cv2.imshow("result", self.panorama)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def __img_position(self):
        homo = np.ones((3, 1))
        homo[0, 0] = self.sift2.img.shape[1] / 2
        homo[1, 0] = self.sift2.img.shape[0] / 2
        new_homo = np.dot(self.__homograph, homo)  # 3x3 · 3x1 = 3x1
        for i in range(new_homo.shape[0]):  # shape[0] = 3
            new_homo[i, 0] = new_homo[i, 0] / new_homo[2, 0]
            # 이렇게 할 경우 new_homo[2,0] 이 1인채로 변한다.
        # 3x3의 호모그래피 행렬을 곱하기 위해 2x1의 shape를 3x1로 늘린 것이기 때문에 다시 2x1로 바꾸기 위한 것
        dx = int(new_homo[0, 0] - homo[0, 0] / 2)  # img2의 사이즈의 절반보다 img2를 변형시킨 사이즈가 더 클경우, (img2가 img1의 왼쪽에 위치하는 경우)
        if dx < 0:  # dx가 0보다 작으면 img2가 img1보다 왼쪽에 위치한다.
            dxt = np.absolute(dx)
            new_h = np.array([[1, 0, dxt], [0, 1, 0], [0, 0, 1]], np.float32)
            homograph = np.dot(new_h, self.__homograph)
        else :
            homograph = self.__homograph
        self.__ip = Imageposition(homograph, dx)

    def __make_trans_img(self):
        dxt = np.absolute(self.__ip.dx)
        y = self.sift1.img.shape[0]
        x = self.sift1.img.shape[1] + dxt
        if self.__ip.dx < 0:
            transformed_image = cv2.warpPerspective(self.sift2.img, self.__ip.h, (x, y))
        else:
            transformed_image = cv2.warpPerspective(self.sift2.img, self.__ip.h,
                                             (self.sift2.img.shape[1] + dxt, self.sift2.img.shape[0]))
        non_transformed_image = np.zeros((y, x, 3), np.uint8)
        self.panorama = np.zeros((y, x, 3), np.uint8)
        for i in range(3):
            if self.__ip.dx < 0:
                non_transformed_image[:, dxt:, i] = self.sift1.img[:, :, i]  # 왼쪽 합성
            else:
                non_transformed_image[:, :self.sift1.img.shape[1], i] = self.sift1.img[:, :self.sift1.img.shape[1], i]  # 오른쪽 합성

        # https://github.com/cynricfu/multi-band-blending/blob/master/multi_band_blending.py
        # multi-band blending 부분 공부하기


class Knnmatcher:
    __matches = None
    # find_homography

    def __init__(self, sift_a, sift_b):
        self.H = None
        self.status = None
        self.__match_keypoints_knn(sift_a.des, sift_b.des, 0.75)
        self.__get_homography(sift_a, sift_b, self.__matches, 3.0)

    def __match_keypoints_knn(self, features_a, features_b, ratio):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw_matches = bf.knnMatch(features_a, features_b, 2)
        self.__matches = []
        for m, n in raw_matches:
            if m.distance < n.distance * ratio:
                self.__matches.append(m)

    def __get_homography(self, sift_a, sift_b, matches, reprojthresh):
        kps_a = np.float32([kp.pt for kp in sift_a.kp])
        kps_b = np.float32([kp.pt for kp in sift_b.kp])
        if len(matches) > 4:
            pts_a = np.float32([kps_a[m.queryIdx] for m in matches])
            pts_b = np.float32([kps_b[m.trainIdx] for m in matches])
            (self.H, self.status) = cv2.findHomography(pts_b, pts_a, cv2.RANSAC,
                                                       reprojthresh)


class SIFT:
    def __init__(self, image_adr):
        self.features = cv2.SIFT_create()
        self.img = cv2.imread(image_adr, cv2.IMREAD_COLOR)
        image_cvt = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.kp, self.des = self.features.detectAndCompute(image_cvt, None)
