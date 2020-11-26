import numpy as np
import cv2

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

# --------------------------------------------------------------


def get_video_second(vidcap, perid_second):

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        hasFrames, image = vidcap.read()

        return hasFrames

    sec = 0
    frameRate = perid_second # Change this number to 1 for each 1 second

    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

# -----------------------------------------------------------------
MIN_MATCHES = 15
detection = False
frameCounter = 0

capVid = cv2.VideoCapture('test_video.mp4')
imgTarget = cv2.imread('target_new.jpg')
myVid = cv2.VideoCapture('video.mp4')

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

# success, imgVideo = myVid.read()
# hT, wT, cT = imgTarget.shape
# imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(imgTarget, None)

while capVid.isOpened():

    success, imgWebcam = capVid.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    # if detection == False:
    #     myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     frameCounter = 0
    # else:
    #     if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
    #         myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #         frameCounter = 0
    #     success, imgVideo = myVid.read()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > MIN_MATCHES:
        detection = True
        imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2,
                                      matches[:MIN_MATCHES], 0, flags=2)

        srcPts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        # print(matrix)

        pts = np.float32([[0,0],[0,hT - 1], [wT - 1, hT - 1], [wT - 1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)

        cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))

        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

        StackedImages = stackImages(([imgWebcam, imgVideo, imgTarget],
                                     [imgFeatures, imgWarp, imgAug]), 0.5)


        # cv2.imshow('imgAug', StackedImages)
        cv2.imshow('imgAug', imgAug)
        # cv2.imshow('imgWarp', imgWarp)
    # cv2.imshow('img2', img2)
    # cv2.imshow('imgFeatures', imgFeatures)
    # cv2.imshow('ImgTarget', imgTarget)
    # cv2.imshow('myVid', imgVideo)
    # cv2.imshow('Webcam', imgWebcam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # frameCounter += 1


# cap.release()
cv2.destroyAllWindows()