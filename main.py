import sift


if __name__ == "__main__":
    # select 2 image_address(example)
    img1 = './foto1A.jpg'
    img2 = './foto1B.jpg'

    pano = sift.PANORAMA(img2, img1)
    pano.show()
