import numpy as np

class headNodCounter:
    def __init__(self):
        self.count = 0
        self.upper = 38
        self.lower = 28

        self.min = 1
        self.max = 15 

        self.pos = 1
        self.prev = [1, 1]

        self.state = 0
        self.count = 0

    def getNpArray(self, landmark):
        return np.array([landmark.x, landmark.y, landmark.z])

    def getHeadAngle(self, landmarks):
        landmarks = landmarks.landmark

        left_shoulder, right_shoulder = self.getNpArray(landmarks[11]), self.getNpArray(landmarks[12]) 
        cross = np.cross(left_shoulder, right_shoulder)

        collar = (left_shoulder + right_shoulder)/2
        nose_vec = self.getNpArray(landmarks[0]) - collar

        nose_vec = nose_vec / np.linalg.norm(nose_vec)
        cross = cross / np.linalg.norm(cross)

        return np.degrees(np.arccos(np.dot(nose_vec, cross)))

    def setPos(self, angle):
        if angle > self.upper:
            curr = 0
        elif angle < self.lower:
            curr = 2
        else:
            curr = 1

        if curr == self.prev[0]:
            self.prev[1] += 1
        else:
            self.prev[0] = curr
            self.prev[1] = 1

        self.pos = curr
        

    def update(self, results):
        landmarks = results.pose_world_landmarks

        if landmarks:
            angle = self.getHeadAngle(landmarks)
            self.setPos(angle)

            if (self.prev[0] != 1) and (self.prev[1] > self.max):
                self.state = 0
            elif (self.state == 0) and (self.pos == 0):
                self.state = 1
            elif (self.state == 1) and (self.pos == 1):
                self.state = 2
            elif (self.state == 2) and (self.pos == 2):
                self.state = 3
            elif (self.state == 3) and (self.pos == 1):
                self.state = 4
            elif (self.state == 4):
                self.state = 0
                self.count += 1

            #print(angle, self.pos)

        return self.count
