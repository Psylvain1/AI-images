#please subscribe "Problem Solve With Ridoy" youtube channel

import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

detector = HandDetector(detectionCon=0.6, maxHands= 4)

while True:
  _, img = cap.read()
  hands, img = detector.findHands(img)
  cv2.imshow("Smart Camera",img)
  if cv2.waitKey(1) == ord("q"):
    break