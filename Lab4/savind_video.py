import numpy as np
import cv2


# Define the codec and create VideoWriter object
# ----------------------------------------
# reading every some period of time

# vidcap = cv2.VideoCapture('video.mp4')
# perid_second = 0.5
#
# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#     hasFrames, image = vidcap.read()
#
#     return hasFrames, image
#
#
# sec = 0
# frameRate = perid_second  # Change this number to 1 for each 1 second
#
# success = getFrame(sec)
# while success:
#     # count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success, img = getFrame(sec)
#     cv2.imshow('img', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# ------------------------------------------
# writing video

# cap = cv2.VideoCapture('video.mp4')
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         frame = cv2.flip(frame,0)
#
#         # write the flipped frame
#         out.write(frame)
#
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# ---------------------------------------
# reading and writing video from some period
#
# vidcap = cv2.VideoCapture('video.mp4')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
#
# perid_second = 0.5
#
# def getFrame(video, sec):
#     video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#     hasFrames, image = video.read()
#
#     return hasFrames, image
#
#
# sec = 0
# frameRate = perid_second  # Change this number to 1 for each 1 second
#
# success, img = getFrame(vidcap, sec)
# img = cv2.flip(img, 0)
# out.write(img)
#
# while success:
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success, img = getFrame(vidcap, sec)
#     img = cv2.flip(img, 0)
#     out.write(img)
#
#     cv2.imshow('img', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# vidcap.release()
# out.release()
# cv2.destroyAllWindows()

#------------------------------------------------
# change video size
import numpy as np

cap = cv2.VideoCapture('test_video.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_video.avi', fourcc, 5, (640, 480))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        out.write(b)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()