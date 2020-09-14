import cv2


# cap = cv2.VideoCapture(0)
#
#
# ret, frame = cap.read()
# cv2.imwrite('webcam.png', frame)
image = cv2.imread('webcam1.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.rectangle(gray, (300, 300), (150, 150), (0, 150, 200), 20)
gray = cv2.line(gray, (100, 150), (400, 200), (100, 200, 255), 5)
cv2.imwrite('webcam_result1.png', gray)

cv2.imshow('gray', gray)
# cv2.imshow('frame', frame)
