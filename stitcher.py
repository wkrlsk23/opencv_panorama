import cv2


class StitcherPano:
    # opencv 에서 기본 으로 제공 하는 Stitcher 객체를 생성
    # 객체 내 함수를 통해 panorama 이미지 제작
    def __int__(self, img1, img2):
        self.img1 = cv2.imread(img1)
        self.img2 = cv2.imread(img2)
        stitcher = cv2.Stitcher_create()
        _, self.pano = stitcher.stitch([img1, img2])

    def show(self):
        cv2.imshow("result", self.pano)
        cv2.waitKey()
        cv2.destroyAllWindows()