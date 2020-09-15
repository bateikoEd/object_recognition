import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('opencv.png', frame)
    image = cv2.imread('opencv.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.rectangle(gray, (100, 100), (200, 200), (0, 255, 255), 10)
    gray = cv2.line(gray, (60, 20), (400, 200), (0, 100, 255), 5)
    cv2.imwrite('opencv2.png', gray)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()