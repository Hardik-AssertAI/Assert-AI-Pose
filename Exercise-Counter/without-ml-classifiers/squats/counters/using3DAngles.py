import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

class squatCounter:
    def __init__(self):
        self.position = 0
        self.state = 0
        self.count = 0
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = mp.solutions.pose.Pose(
                        min_detection_confidence = 0.5,
                        min_tracking_confidence = 0.5
                        )
        
    def getAngleBetweenVectors(self, v1, v2):
        v1 = v1/ np.linalg.norm(v1)
        v2 = v2/ np.linalg.norm(v2)

        return np.degrees(np.arccos(np.dot(v1, v2)))

    def getLegAngle(self, leg_landmarks):
        v1 = np.array([
                (leg_landmarks[0].x - leg_landmarks[1].x),
                (leg_landmarks[0].y - leg_landmarks[1].y),
                (leg_landmarks[0].z - leg_landmarks[1].z)
                ])

        v2 = np.array([
                (leg_landmarks[2].x - leg_landmarks[1].x),
                (leg_landmarks[2].y - leg_landmarks[1].y),
                (leg_landmarks[2].z - leg_landmarks[1].z)
                ])

        return self.getAngleBetweenVectors(v1, v2)

    def makeUpdates(self, left, right):
        if left <= 100 and right <= 100:
            currPosition = 2

            if self.state == 1:
                self.count += 1

            self.state = 2
        elif left >= 140 and right >= 140:
            currPosition = 0

            self.state = 0
        elif left <= 140 and right <= 140:
            currPosition = 1

            if self.state == 0:
                self.state = 1
        else:
            currPosition = -1

        self.position = currPosition 

    def feedFrame(self, image, show = True):
        # resize
        # image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
        height, width, _ = image.shape

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        left_leg = [results.pose_world_landmarks.landmark[i] for i in [23, 25, 29]]
        right_leg = [results.pose_world_landmarks.landmark[i] for i in [24, 26, 30]]
        
        leftLegAngle = self.getLegAngle(left_leg)
        rightLegAngle = self.getLegAngle(right_leg)

        left_show_leg = [results.pose_landmarks.landmark[i] for i in [23, 25, 29]]
        right_show_leg = [results.pose_landmarks.landmark[i] for i in [24, 26, 30]]

        landmark_subset = landmark_pb2.NormalizedLandmarkList(
            landmark = left_show_leg + right_show_leg
        )
        
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.mp_drawing.draw_landmarks(
            image,
            landmark_subset
            )

        self.makeUpdates(leftLegAngle, rightLegAngle)
        #print(leftLegAngle, rightLegAngle, self.state)

        label = f'count = {self.count}'
        cv2.putText(image, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA) 

        if show:
            cv2.imshow('MediaPipe Pose', image)

        return image
