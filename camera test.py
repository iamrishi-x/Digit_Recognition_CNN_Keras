import cv2

width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3,width)  #3 for width
cap.set(4,height) #4 for height

while True:
    success, imgOriginal = cap.read()
    cv2.imshow('img',imgOriginal)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()