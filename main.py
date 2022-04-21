import pano
import cv2

if __name__ == "__main__":
    # select 2 image_address(example)
    img1 = './foto1A.jpg'
    img2 = './foto1B.jpg'
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    stitcher = cv2.Stitcher_create()
    _, dst = stitcher.stitch([img1, img2])
    cv2.imshow("result",dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    #pano = pano.PANORAMA(img2, img1)
    #pano.show()