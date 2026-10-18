import mediapipe as mp
import cv2

img= cv2.imread("/home/ssppa/image_learning/people.jpg")
img = cv2.resize(img,(600,600), interpolation=cv2.INTER_NEAREST)

"""
cv2.imshow("Resized Image", img)  # Add a window title here
cv2.waitKey(0)  # Keeps the window open until a key is pressed
cv2.destroyAllWindows()  # Closes all OpenCV windows
"""
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    annotated_image = img.copy()
    print("Number of people detected",len(results.detections), "\n\n")
    print("Description of people \n")
    for i, detection in enumerate(results.detections):
        print("Person",i,"\n")
        print("Confidence score", detection.score, "\n")
        box = detection.location_data.relative_bounding_box
        x_start, y_start = int(box.xmin*img.shape[1]),int(box.ymin*img.shape[0])
        x_end, y_end = int((box.xmin+box.width)*img.shape[1]),int((box.ymin + box.height)*img.shape[0])
        annotated_image=cv2.rectangle(img,(x_start,y_start),(x_end,y_end),(0,255,0),5)
        mp_drawing.draw_detection(annotated_image,detection)
    cv2.imshow("Resized Image", annotated_image)  # Add a window title here
    cv2.waitKey(0)  # Keeps the window open until a key is pressed
    cv2.destroyAllWindows()  # Closes all OpenCV windows
     
