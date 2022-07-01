import pickle
import sys
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

model = pickle.load(open('knnmodel.pkl', 'rb'))
classes = ['standing', 'squatting', 'middle']

state = -1
count = 0

cap = cv2.VideoCapture(sys.argv[1])
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

writer = cv2.VideoWriter("test1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

    while True:      
        ret, frame = cap.read()

        if not ret:
            break

        #frame = cv2.resize(frame, (0,0), fx=1, fy=1)
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
            if clss == 0:
                state = 0
            elif clss == 1 and state == 0:
                state = 1 
                count += 1

            label = f"count = {count}, {classes[clss[0]]}"

            cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA) 
            
                
        writer.write(frame)
cap.release()
cv2.destroyAllWindows()
