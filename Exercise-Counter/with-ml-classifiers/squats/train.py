import cv2
import os
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import pickle 

mp_pose = mp.solutions.pose

X = []
y = []
def extractFeaturesFromDataset(folder, clss):
    global X, y

    files = os.listdir(folder)
    i = 0
    for file in files:
        image = cv2.imread(os.path.join(folder, file))

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = results.pose_world_landmarks

        if landmarks:
            landmarks = [landmarks.landmark[i] for i in [11, 12, 23, 24, 25, 26, 27, 28]] 
            x = []
            for landmark in landmarks:
                x.append(landmark.x)
                x.append(landmark.y)
                x.append(landmark.z)

            X.append(x)
            y.append(clss)
        else:
            print(f"bruh {file}")


        print(f"{i}/{len(files)}")
        i += 1

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
    
    extractFeaturesFromDataset("./dataset/standing/", 0)
    extractFeaturesFromDataset("./dataset/squatting/", 1)
    extractFeaturesFromDataset("./dataset/middle/", 2)
    print("Finished extracting features...")

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, y)
    
    with open("knnmodel5.pkl", "wb") as f:
        pickle.dump(neigh, f)
    

