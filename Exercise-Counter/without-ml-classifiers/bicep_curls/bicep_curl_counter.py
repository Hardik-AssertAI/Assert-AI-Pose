import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def getAngle(shoulder,elbow,wrist): 

    '''Returns the angle between the upper and lower arm, formed by the shoulder, elbow and wrist as a triplet.
    '''
    shoulder = np.array([shoulder.x, shoulder.y])
    elbow = np.array([elbow.x, elbow.y])
    wrist = np.array([wrist.x, wrist.y])

    upper_arm = np.subtract(shoulder,elbow)
    lower_arm = np.subtract(elbow,wrist)
    
    theta = np.arccos(np.dot(upper_arm,lower_arm) / np.multiply(np.linalg.norm(lower_arm), np.linalg.norm(upper_arm)))     # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = 180 - 180 * theta /np.pi 
      
    return theta


def countBicepCurls(video_path):

    video = cv2.VideoCapture(video_path)

    annotated_frames = []
    right_count=0
    left_count=0
    flag_right = None
    flag_left = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            # BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      
            image.flags.writeable = False
            
            # Make Detections
            results = pose.process(image)                       

            # Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)     

        
            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]


                # Calculate angle
            theta_right = getAngle(right_shoulder,right_elbow,right_wrist)
            theta_left = getAngle(left_shoulder,left_elbow,left_wrist)
            #print(f'Right angle: {theta}')

                # Visualize angle
            #cv2.putText(image,\
                    #str(theta), \
                        # tuple(np.multiply([right_elbow.x, right_elbow.y], [640,480]).astype(int)),\
                            # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
            
                # Counter 
            if theta_right > 170:
                flag_right = 'down'
            if theta_right < 50 and flag_right=='down':
                right_count += 1
                print('Right count incremented')
                flag_right = 'up'
            if theta_left > 170:
                flag_left = 'down'
            if theta_left < 50 and flag_left=='down':
                left_count += 1
                print('Left count incremented')
                flag_left = 'up'
                


            # Setup Status Box
            #cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            cv2.putText(image, str(right_count+left_count), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)


            # Render Detections
            landmark_subset = mp.framework.formats.landmark_pb2.NormalizedLandmarkList(landmark = [results.pose_landmarks.landmark[i] for i in [11,12,13,14,15,16]])
            mp_drawing.draw_landmarks(image, landmark_subset)

            annotated_frames.append(image)

            cv2.imshow('Annotated',image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    
    height,width,layers = annotated_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    FPS = video.get(cv2.CAP_PROP_FPS)
    annotated_video=cv2.VideoWriter('bicep_curl_ninad.mp4',fourcc,FPS,(width,height))

    for i in range(len(annotated_frames)):
       annotated_video.write(annotated_frames[i])

    print('Success')

    cv2.destroyAllWindows()
    video.release()
    annotated_video.release()

def main():
    video_path = 0
    countBicepCurls(video_path)

main()






