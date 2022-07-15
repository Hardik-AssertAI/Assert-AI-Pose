import torch
import cv2
import sys
import mediapipe as mp
from headNodCounter import headNodCounter
import pickle

mp_pose = mp.solutions.pose

nodCounter = headNodCounter()
nod_count = 0

namaste = pickle.load(open('models/svcnamaste.pkl', 'rb'))
hibye = pickle.load(open('models/svchibye2.pkl', 'rb'))

vid = cv2.VideoCapture(sys.argv[1])
#frame_width = int(vid.get(3))
#frame_height = int(vid.get(4))
#size = (frame_width, frame_height)

#writer = cv2.VideoWriter("bruh.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    while True:      
        ret, frame = vid.read()

        if not ret:
            break

        #frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
        frame_rgb = frame[:, :, ::-1]
        results = pose.process(frame_rgb)
        landmarks = results.pose_world_landmarks


        if landmarks: 

            landmarks1 = [landmarks.landmark[i] for i in range(11, 23)] 
            x = []
            for landmark in landmarks1:
                x.append(landmark.x)
                x.append(landmark.y)
                x.append(landmark.z)

            pred1 = namaste.predict([x])[0]
            pred2 = hibye.predict([x])[0]
            
            if pred1 == 0:
                cv2.putText(frame, "namaste", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            elif pred2 == 0:
                cv2.putText(frame, "hi/bye", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                nod_count = nodCounter.update(results)

        #cv2.putText(frame, f"nod_count : {nod_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            
        #writer.write(frame)
        cv2.imshow("no", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):    
            break 

vid.release()
cv2.destroyAllWindows()
