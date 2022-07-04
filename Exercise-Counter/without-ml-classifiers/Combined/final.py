# from google.colab.patches import cv2_imshow
import glob
from PIL import Image
from mediapipe.framework.formats import landmark_pb2
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calc_angle(a, b, c):  # 3D points
    ''' Arguments:
        a,b,c -- Values (x,y,z, visibility) of the three points a, b and c which will be used to calculate the
                vectors ab and bc where 'b' will be 'elbow', 'a' will be shoulder and 'c' will be wrist.
        
        Returns:
        theta : Angle in degress between the lines joined by coordinates (a,b) and (b,c)
    '''
    a = np.array([a.x, a.y])  # , a.z])    # Reduce 3D point to 2D
    b = np.array([b.x, b.y])  # , b.z])    # Reduce 3D point to 2D
    c = np.array([c.x, c.y])  # , c.z])    # Reduce 3D point to 2D

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)

    # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = np.arccos(
        np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))
    theta = 180 - 180 * theta / 3.14    # Convert radians to degrees
    return np.round(theta, 2)


def distance(a, b):
    a = np.array([a.x, a.y])  # , a.z])    # Reduce 3D point to 2D
    b = np.array([b.x, b.y])  # , b.z])    # Reduce 3D point to 2D

    ab = np.subtract(a, b)

    return np.linalg.norm(ab)


def getAngleBetweenVectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    return np.degrees(np.arccos(np.dot(v1, v2)))


def getLegAngle(leg_landmarks):
    v1 = np.array([
        (leg_landmarks[0].x - leg_landmarks[1].x),
        (leg_landmarks[0].y - leg_landmarks[1].y),
        (leg_landmarks[0].z - leg_landmarks[1].z)])

    v2 = np.array([
        (leg_landmarks[2].x - leg_landmarks[1].x),
        (leg_landmarks[2].y - leg_landmarks[1].y),
        (leg_landmarks[2].z - leg_landmarks[1].z)])

    return getAngleBetweenVectors(v1, v2)

def getAngle(shoulder,elbow,wrist): 
    '''Returns the angle between the upper and lower arm, formed by the shoulder, elbow and wrist as a triplet.
    '''
    shoulder = np.array([shoulder.x, shoulder.y])
    elbow = np.array([elbow.x, elbow.y])
    wrist = np.array([wrist.x, wrist.y])

    upper_arm = np.subtract(shoulder,elbow)
    lower_arm = np.subtract(elbow,wrist)
    
    theta = np.arccos(np.dot(upper_arm,lower_arm) / np.multiply(np.linalg.norm(lower_arm), np.linalg.norm(upper_arm)))     # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = 180 - 180 * theta / 3.14 
      
    return theta

frames = []
counter_both_hands = 0
counter_one_hand = 0
counter_squats = 0
counter_bicep_curl_right = 0
counter_bicep_curl_left = 0

flag_both_hands = None
flag_bicep_curl_left = None
flag_bicep_curl_right = None


b1 = False
b2 = False

position = 0
state = 0


cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
#print(fps)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success: 
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
    )
    # Flip the image horizontally for a selfie-view display.
    #image=cv2.resize(image,(1080,720))
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

    d_left1 = distance(left_hip, left_elbow)
    d_right1 = distance(right_hip, right_elbow)

    d_left2 = distance(left_shoulder, left_hip)
    d_right2 = distance(right_shoulder, right_hip)

    ###############################################################################

    if d_left1 > d_left2 + 0.05 and d_right1 > d_right2 + 0.05:
        flag_both_hands = 'up'
    if d_left1 + 0.05 < d_left2 and d_right1 + 0.05 < d_right2 and flag_both_hands == 'up':
        counter_both_hands += 1
        flag_both_hands = 'down'

    ##############################################################################

    theta_right = getAngle(right_shoulder, right_elbow, right_wrist)
    theta_left = getAngle(left_shoulder, left_elbow, left_wrist)
    
        
    if theta_right > 160:
        flag_bicep_curl_right = 'down'
    if theta_right < 40 and flag_bicep_curl_right=='down':
        counter_bicep_curl_right += 1
        print('Right count incremented')
        flag_bicep_curl_right = 'up'

    
    if theta_left > 160:
        flag_bicep_curl_left = 'down'
    if theta_left < 40 and flag_bicep_curl_left=='down':
        counter_bicep_curl_left+= 1
        print('Left count incremented')
        flag_bicep_curl_left = 'up'

    ###############################################################################

    image_height, image_width, _ = image.shape

    left_index_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * image_height
    left_pinky_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y * image_height
    left_thumb_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y * image_height
    left_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height
    left_elbow_y = results.pose_landmarks.landmark[13].y * image_height
    left_hip_y = results.pose_landmarks.landmark[23].y * image_height
    left_shoulder_y = results.pose_landmarks.landmark[11].y * image_height

    right_index_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * image_height
    right_pinky_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y * image_height
    right_thumb_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y * image_height
    right_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height
    right_elbow_y = results.pose_landmarks.landmark[14].y * image_height
    right_hip_y = results.pose_landmarks.landmark[24].y * image_height
    right_shoulder_y = results.pose_landmarks.landmark[12].y * image_height

    if(left_elbow_y > left_shoulder_y and left_wrist_y > left_shoulder_y):
      b1 = False
    if(right_elbow_y > right_shoulder_y and right_wrist_y > right_shoulder_y):
      b2 = False

    if(left_elbow_y < left_shoulder_y and left_wrist_y < left_shoulder_y):
      if((not b1) and (not b2)):
        b1 = True
        if(not(abs(right_elbow_y-right_shoulder_y) < (image_height//20))):
          counter_one_hand += 1

    if(right_elbow_y < right_shoulder_y and right_wrist_y < right_shoulder_y):
      if((not b1) and (not b2)):
        b2 = True
        if(not(abs(left_elbow_y-left_shoulder_y) < (image_height//20))):
          counter_one_hand += 1
    
    ###############################################################################

    left_leg = [results.pose_world_landmarks.landmark[i] for i in [23, 25, 29]]
    right_leg = [results.pose_world_landmarks.landmark[i]
                 for i in [24, 26, 30]]

    leftLegAngle = getLegAngle(left_leg)
    rightLegAngle = getLegAngle(right_leg)

    if leftLegAngle <= 100 and rightLegAngle <= 100:
        currPosition = 2
        if state == 1:
            counter_squats += 1
        state = 2
    elif leftLegAngle >= 140 and rightLegAngle >= 140:
        currPosition = 0
        state = 0
    elif leftLegAngle <= 140 and rightLegAngle <= 140:
        currPosition = 1
        if state == 0:
            state = 1
    else:
        currPosition = -1
    position = currPosition


#######################################################################################

    # image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

    #print(image.shape)

    outp = "One hand raises = "+str(counter_one_hand)
    image = cv2.rectangle(image, (5, 10), (185, 40), (255, 255, 255), -1)
    image = cv2.putText(image, outp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)

    
    outp1 = "Both hand raises = "+str(counter_both_hands)
    image = cv2.rectangle(
        image, (5, 50), (190, 80), (255, 255, 255), -1)
    cv2.putText(image, outp1, (10, 70),cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # SQUATS TEXT
    outp2 = "Squats = "+str(counter_squats)
    image = cv2.rectangle(
        image, (image_width-115, 10), (image_width-5, 40), (255, 255, 255), -1)
    cv2.putText(image, outp2, (image_width-105, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)

    outp3 = "Bicep_curls = "+str(counter_bicep_curl_left+counter_bicep_curl_right)
    image = cv2.rectangle(
        image, (image_width-145, 50), (image_width-5, 80), (255, 255, 255), -1)
    cv2.putText(image, outp3, (image_width-135, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('MediaPipe Pose',image)
    frames.append(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

  #height, width, depth = frames[0].shape
  #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  #annotated_video = cv2.VideoWriter('annotated_ninad.mp4', fourcc, 30, (width, height))

  #for i in range(len(frames)):
   # annotated_video.write(frames[i])

print("frames",len(frames))

print("Both hands", counter_both_hands)
print("One hand", counter_one_hand)
print("Squats", counter_squats)
print("bicep curl", counter_bicep_curl_left + counter_bicep_curl_right)

cap.release()
#annotated_video.release()