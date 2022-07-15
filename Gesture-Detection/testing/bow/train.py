import cv2
import os
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
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
            landmarks = [landmarks.landmark[i] for i in [11, 12, 23, 24, 25, 26]] 
            x = []
            for landmark in landmarks:
                x.append(landmark.x)
                x.append(landmark.y)
                x.append(landmark.z)

            X.append(x)
            y.append(clss)
        else:
            print("bruh")


        print(f"{i}/{len(files)}")
        i += 1

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
    
    extractFeaturesFromDataset("./dataset/bow/", 0)
    extractFeaturesFromDataset("./dataset/no-bow/", 1)
    print("Finished extracting features...")

    print("Started training")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    with open("models/knnmodel.pkl", "wb") as f:
        pickle.dump(knn, f)

    svc = make_pipeline(StandardScaler(), SVC(gamma='auto', probability = True))
    svc.fit(X, y)
    with open("models/svcmodel.pkl", "wb") as f:
        pickle.dump(svc, f)

    dtc = DecisionTreeClassifier(max_depth = 3, random_state = 0)
    dtc.fit(X, y)
    with open("models/dtcmodel.pkl", "wb") as f:
        pickle.dump(dtc, f)

    rfc = RandomForestClassifier(max_depth = 3, random_state = 0)
    rfc.fit(X, y)
    with open("models/rfcmodel.pkl", "wb") as f:
        pickle.dump(rfc, f)
    
    gnb = GaussianNB()
    gnb.fit(X, y)
    with open("models/gnbmodel.pkl", "wb") as f:
        pickle.dump(gnb, f)

