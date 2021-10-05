import mediapipe as mp #imports a drawing utilities wich will help us to draw utils from specific holistic module
import cv2 #import a holistic module

mp_drawing=mp.solutions.drawing_utils;
mp_holistic=mp.solutions.holistic;

cap=cv2.VideoCapture(0)
# Inisiate Holistic Module
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame=cap.read();
        #Recolor Feed
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);
        #Make Detection
        result=holistic.process(image);
        #print(result.face_landmarks)
        
        # face_landmarks pose_landmarks left_hand_landmarks right_hand_landmarks 
        # Recoloring Back to BGR
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR);
        
        #Draw Face Landmarks
        mp_drawing.draw_landmarks(image,result.face_landmarks,mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2));
        
        #Draw Right Hand
        mp_drawing.draw_landmarks(image,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2));
        
        #Draw Left Hand
        mp_drawing.draw_landmarks(image,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(255,0,0),thickness=2,circle_radius=2));
        
        # Draw pose
        mp_drawing.draw_landmarks(image,result.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(81,100,255),thickness=2,circle_radius=2));
        
        
        
        cv2.imshow('Body detection',image);
    
        if (cv2.waitKey(10) & 0xFF == ord('e')):
            break;
    
    
    cap.release();
    cv2.destroyAllWindows();
    

