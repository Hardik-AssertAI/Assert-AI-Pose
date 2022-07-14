import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np
from mediapipe.framework.formats import landmark_pb2

def calc_angle(a,b,c): # 3D points
    ''' Arguments:
        a,b,c -- Values (x,y,z, visibility) of the three points a, b and c which will be used to calculate the
                vectors ab and bc where 'b' will be 'elbow', 'a' will be shoulder and 'c' will be wrist.
        
        Returns:
        theta : Angle in degress between the lines joined by coordinates (a,b) and (b,c)
    '''
    a = np.array([a.x, a.y])#, a.z])    # Reduce 3D point to 2D
    b = np.array([b.x, b.y])#, b.z])    # Reduce 3D point to 2D
    c = np.array([c.x, c.y])#, c.z])    # Reduce 3D point to 2D

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)
    
    theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))     # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = 180 - 180 * theta / 3.14    # Convert radians to degrees
    return np.round(theta, 2)


def distance(a, b):
    a = np.array([a.x, a.y])#, a.z])    # Reduce 3D point to 2D
    b = np.array([b.x, b.y])#, b.z])    # Reduce 3D point to 2D

    ab = np.subtract(a, b)

    return np.linalg.norm(ab)



flag = None
frames = []
counter = 0
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue;

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    landmark_subset = landmark_pb2.NormalizedLandmarkList(
      landmark = [
          results.pose_landmarks.landmark[11],
          results.pose_landmarks.landmark[12],
          results.pose_landmarks.landmark[15],
          results.pose_landmarks.landmark[16], 
          results.pose_landmarks.landmark[23],
          results.pose_landmarks.landmark[24], 
      ]
      )
    
    mp_drawing.draw_landmarks(
        image,
        landmark_subset,
        )
    # Flip the image horizontally for a selfie-view display.
    #mage=cv2.resize(image,(1080,720))
    #cv2_imshow(image)

    landmarks = results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]

    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    #angle_left = calc_angle(left_wrist, left_shoulder, left_hip)      #  Get angle
    #vangle_right = calc_angle(right_wrist, right_shoulder, right_hip)      #  Get angle 

    #d_left = distance(left_hip, left_wrist)
    #d_right = distance(right_hip, right_wrist)

    d_left1 = distance(left_hip, left_elbow)
    d_right1 = distance(right_hip, right_elbow)

    d_left2 = distance(left_shoulder, left_hip)
    d_right2 = distance(right_shoulder, right_hip)

    if d_left1 > d_left2 + 0.05 and d_right1 > d_right2 + 0.05:
        flag = 'up'
    if d_left1 + 0.05 < d_left2 and d_right1 + 0.05 < d_right2  and flag == 'up':
        counter += 1
        flag = 'down'
    #image = cv2.putText(image, counter, position, font, fontScale, fontColor)
    #image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

    #print(image.shape)

    # Setup Status Box
    #cv2.rectangle(image, (0,252), (100,352), (245,117,16), 2)
    cv2.putText(image, str(counter), (580,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
    #image = cv2.flip(image, 1)
    frames.append(image)
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

  height,width, depth = frames[0].shape
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  annotated_video=cv2.VideoWriter('final_video.mp4',fourcc, 25 ,(width,height))

  for i in range(len(frames)):
    annotated_video.write(frames[i])

  print(len(frames))
cap.release()
annotated_video.release()


