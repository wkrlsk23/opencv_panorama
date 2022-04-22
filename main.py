import pano
import stitcher

if __name__ == "__main__":
    # select 2 image_address(example)
    img1 = './foto1A.jpg'
    img2 = './foto1B.jpg'
    # 원하는 방법에 맞춰 아래 변수를 조절 하시오.
    case = 2

    if case == 1:
        # 1번 방법. Opencv 내 Stitcher 함수 사용 하기
        pano = stitcher.StitcherPano(img1, img2)
        pano.show()
    elif case == 2:
        # 2번 방법. Panorama 구현 하기
        pano = pano.PANORAMA(img1, img2)
        pano.show()
