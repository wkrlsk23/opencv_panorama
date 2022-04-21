import numpy as np
import cv2


class Knnmatcher:
    __matches = None

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
