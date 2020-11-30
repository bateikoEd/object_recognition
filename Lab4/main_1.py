import cv2
import numpy as np
from PIL import Image

def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW,sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

MIN_MATCHES = 20
detection = False
frameCounter = 0

capVid = cv2.VideoCapture('video/test_vertical.mp4')

model = cv2.imread('images/Target_resize_1.jpg')
myVid = cv2.VideoCapture('video/video.mp4')
# success, imgVideo = myVid.read()

# resize imgVideo
hT, wT, cT = model.shape

#resize imgFrame
# imgVideo = image_resize(imgVideo, width=wT, height=hT)

# detection = False
# frameCounter = 0

orb = cv2.ORB_create()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp_model, des_model = orb.detectAndCompute(model, None)


while capVid.isOpened() and myVid.isOpened():


    _, cap = capVid.read()
    cap = image_resize(cap, width=hT, height=wT)
    imgAug = cap.copy()

    kp_frame, des_frame = orb.detectAndCompute(cap, None)
    matches = bf.match(des_model, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    # video frames
    _, imgVideo = myVid.read()
    imgVideo = image_resize(imgVideo, width=wT, height=hT)

    if len(matches) > MIN_MATCHES:
        # detection = True
        # draw first 15 matches.
        imgFeatures = cv2.drawMatches(model, kp_model, cap, kp_frame,
                              matches[:MIN_MATCHES], 0, flags=2)

        print('\n\n\n\n--------------------\n\n\n\n')
        print(f'kp1_len:\t{len(kp_model)}')
        print(f'kp2_len:\t{len(kp_frame)}')
        print(f'matches_len:\t{len(matches)}')


        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # compute Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Draw a rectangle that marks the found model in the frame
        h, w, _ = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, M)
        # connect them with lines
        img2 = cv2.polylines(cap, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        imgWarp = cv2.warpPerspective(imgVideo, M, (cap.shape[1], cap.shape[0]))
        maskNew = np.zeros((cap.shape[0], cap.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))

        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

        StackedImages = stackImages(([cap, imgVideo, model],
                                     [imgFeatures, imgWarp, imgAug]), 0.5)


        cv2.imshow('StackedImages', StackedImages)
        # cv2.imwrite('result/aug.jpg', StackedImages)
        # cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # frameCounter += 1

cv2.destroyAllWindows()