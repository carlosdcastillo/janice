import cv2


def variance_of_laplacian(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    return cv2.Laplacian(lab_planes[0], cv2.CV_64F).var()


def main():
    ksize = 3
    for fn in ['michelle_rodriguez_0001.jpg', 'michelle_rodriguez_0002.jpg']:
        print '--------------'
        im = cv2.imread(fn, cv2.IMREAD_COLOR)
        for i in range(5):
            print 'blur level:', i, 'variance of laplacian:', variance_of_laplacian(im)
            im = cv2.blur(im, (ksize, ksize))


if __name__ == "__main__":
    main()
