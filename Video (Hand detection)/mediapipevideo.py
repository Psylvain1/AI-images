import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


with mp_face_detection.FaceDetection(
    model_selection = 0, min_detection_confidence=0.5) as face_detection:
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == False:
            break
        temp_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_detection.process(temp_frame)
        if results.detections:
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    x_start, y_start = int(box.xmin*frame.shape[1]),int(box.ymin*frame.shape[0])
                    x_end, y_end = int((box.xmin+box.width)*frame.shape[1]),int((box.ymin + box.height)*frame.shape[0])
                    annotated_image=cv2.rectangle(frame,(x_start,y_start),(x_end,y_end),(0,255,0),5)
                    score = str(detection.score)[3:5]+ "%" 
                    txt_coords = (x_start-20,y_start-20)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1
                    txt_col = (255,0,0)
                    thickness = 2
                    cv2.putText(annotated_image,score,txt_coords,font,scale,txt_col,thickness,cv2.LINE_AA)
                cv2.imshow("fin", annotated_image)
                if cv2.waitKey(25) & 0xFF==ord('q'):
                    break
cap.release()
cv2.destroyAllWindows()