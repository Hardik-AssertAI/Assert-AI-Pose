import pickle
import sys
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

knn = pickle.load(open('models/knnmodel.pkl', 'rb'))
dtc = pickle.load(open('models/dtcmodel.pkl', 'rb'))
gnb = pickle.load(open('models/gnbmodel.pkl', 'rb'))
rfc = pickle.load(open('models/rfcmodel.pkl', 'rb'))
svc = pickle.load(open('models/svcmodel.pkl', 'rb'))

classes = ['standing', 'sitting', 'lying', 'hands_up']

cap = cv2.VideoCapture(sys.argv[1])
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    while True:      
        ret, frame = cap.read()

        if not ret:
            break

        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = results.pose_world_landmarks

        if landmarks:
            landmarks = [landmarks.landmark[i] for i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]] 
            x = []
            for landmark in landmarks:
                x.append(landmark.x)
                x.append(landmark.y)
                x.append(landmark.z)

            knn_pred = knn.predict([x]) 
            gnb_pred = gnb.predict([x])
            dtc_pred = dtc.predict([x])       
            rfc_pred = rfc.predict([x]) 
            svc_pred = svc.predict([x])     
                            
            knn_pred = [classes[i] for i in knn_pred]
            gnb_pred = [classes[i] for i in gnb_pred]
            dtc_pred = [classes[i] for i in dtc_pred]
            rfc_pred = [classes[i] for i in rfc_pred]
            svc_pred = [classes[i] for i in svc_pred]

            label1 = f"KNN:{','.join(knn_pred)}"
            label2 = f"GNB:{','.join(gnb_pred)}" 
            label3 = f"DTC:{','.join(dtc_pred)}" 
            label4 = f"RFC:{','.join(rfc_pred)}" 
            label5 = f"SVC:{','.join(svc_pred)}"

            cv2.putText(frame, label1, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 
            cv2.putText(frame, label2, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 
            cv2.putText(frame, label3, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 
            cv2.putText(frame, label4, (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 
            cv2.putText(frame, label5, (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) 
            

        cv2.imshow('nice', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):    
            break 

cap.release()
cv2.destroyAllWindows()
