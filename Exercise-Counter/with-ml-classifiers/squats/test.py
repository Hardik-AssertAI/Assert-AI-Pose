import pickle
import sys
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

model = pickle.load(open('knnmodel.pkl', 'rb'))
classes = ['standing', 'squatting', 'middle']

vid = cv2.VideoCapture(sys.argv[1])
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    while True:      
        ret, frame = vid.read()

        if not ret:
            break

        frame = cv2.resize(frame, (0,0), fx=0.2, fy=0.2)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = results.pose_world_landmarks

        if landmarks:
            landmarks = [landmarks.landmark[i] for i in [11, 12, 23, 24, 25, 26, 27, 28]] 
            x = []
            for landmark in landmarks:
                x.append(landmark.x)
                x.append(landmark.y)
                x.append(landmark.z)

            clss = model.predict([x])
            label = classes[clss[0]]

            cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA) 
            
                
        cv2.imshow('nice', frame)
        if cv2.waitKey(1000//30) & 0xFF == ord('q'):    
            break 

vid.release()
cv2.destroyAllWindows()
