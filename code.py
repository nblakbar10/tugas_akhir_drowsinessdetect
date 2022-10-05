#!/usr/bin/env python

import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import pygame 
import matplotlib.pyplot as plt

#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("cascade.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
predictor_path = ("shape_predictor.dat")

#constanta for 3d face marking
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle

#Calculating  EAR and MAR
def eye_aspect_ratio(eye):
    #compute the euclidean distances between the vertical
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
     # compute the euclidean distance between the horizontal
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    #compute the euclidean distances between the vertical
    A = dist.euclidean(mouth[5], mouth[8])
    B = dist.euclidean(mouth[1], mouth[11])
     # compute the euclidean distance between the horizontal
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C) 

#Set Constant thershold values for EAR and MAR
# EAR_THRESH = 0.25
# EAR_CONSEC_FRAMES = 20
# MAR_THRESH = 0.5

# PC :
# EAR_THRESH = 0.33 #0.3
# MAR_THRESH = 0.5 #0.45 #0.9
# EAR_CONSEC_FRAMES = 18
#raspi :
EAR_THRESH = 0.32 #0.3
MAR_THRESH = 0.43 #0.45 #0.9
EAR_CONSEC_FRAMES = 11.5

#Initialize variables and Counter values
X1 = []
ear = 0
X2 = []
mar = 0
COUNTER_FRAMES_EYE = 0
COUNTER_FRAMES_MOUTH = 0
COUNTER_BLINK = 0
COUNTER_MOUTH = 0

#Initializing Camera for Video Feed
videoSteam = cv2.VideoCapture(0)
ret, frame = videoSteam.read()
size = frame.shape

#Initialize dlib's face detector and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()    ###################################
predictor = dlib.shape_predictor(predictor_path)

#Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#Indexes for calculating the head position of the driver
model_points = np.array([(0.0, 0.0, 0.0),
                         (0.0, -330.0, -65.0),        
                         (-225.0, 170.0, -135.0),     
                         (225.0, 170.0, -135.0),      
                         (-150.0, -150.0, -125.0),    
                         (150.0, -150.0, -125.0)])

focal_length = size[1]
center = (size[1]/2, size[0]/2)

camera_matrix = np.array([[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")

dist_coeffs = np.zeros((4,1))
t_end = time.time()

#Creating a loop for capturing the video
while(True):

    #Grab the frame from the camera, resize it, and convert it to grayscale
    ret, frame = videoSteam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

   #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    for rect in rects:
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        reprojectdst, euler_angle = get_head_pose(shape)

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye) 
        ear = (leftEAR + rightEAR) / 2.0 #calculate eye aspect ratio
        mar = mouth_aspect_ratio(mouth)
        X1.append(ear)
        X2.append(mar)
        image_points = np.array([
                                (shape[30][0], shape[30][1]),
                                (shape[8][0], shape[8][1]),
                                (shape[36][0], shape[36][1]),
                                (shape[45][0], shape[45][1]),
                                (shape[48][0], shape[48][1]),
                                (shape[54][0], shape[54][1])
                                ], dtype="double")


        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], 0, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], 0, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], 0, (255, 255, 255), 1)
        cv2.line(frame, p1, p2, (255,255,255), 2)

        xz=euler_angle[2,0]

        if(xz>10) or (xz<-10):
            # print("right")
            cv2.putText(frame, "PERINGATAN TERLALU MENUNDUK KESAMPING!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), thickness=2)
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()

        if p2[1] > p1[1]*1.4 : #from 1.5
            pygame.mixer.music.play() #INI BUAT PLAY ALARM
            cv2.putText(frame, "PERINGATAN TERLALU MENUNDUK!", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if ear < EAR_THRESH:
            COUNTER_FRAMES_EYE += 1
            if COUNTER_FRAMES_EYE >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "PERINGATAN MATA MENGANTUK!", (150, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pygame.mixer.music.play() #INI BUAT PLAY ALARM
        else:
            if COUNTER_FRAMES_EYE > 2:
                COUNTER_BLINK += 1
                pygame.mixer.music.stop()
            COUNTER_FRAMES_EYE = 0
            # ALARM_ON = False
        
        if mar >= MAR_THRESH:
            cv2.putText(frame, "PERINGATAN MENGUAP!", (150, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            pygame.mixer.music.play() #INI BUAT PLAY ALARM
            COUNTER_FRAMES_MOUTH += 1
        else:
            if COUNTER_FRAMES_MOUTH > 5:
                COUNTER_MOUTH += 1
                pygame.mixer.music.stop()
            COUNTER_FRAMES_MOUTH = 0
            # ALARM_ON = False
        
        if (time.time() - t_end) > 60:
            t_end = time.time()
            COUNTER_BLINK = 0
            COUNTER_MOUTH = 0
            
        cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (500, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (500, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (500, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)

        cv2.putText(frame, "RASIO MATA: {:.2f}".format(ear), (30, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "RASIO MULUT: {:.2f}".format(mar), (240, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    time.sleep(0.02)

fig = plt.figure()
plt.ylabel('EAR Values')
plt.xlabel('Time in Seconds')
ex = plt.subplot(111)
ex.plot(X1)
plt.title('EAR Graph')
ex.legend()
fig.savefig('EAR_Graph.png')

fig = plt.figure()
plt.ylabel('MAR Values')
plt.xlabel('Time in Seconds')
mx = plt.subplot(111)
mx.plot(X2)
plt.title('MAR Graph')
mx.legend()
fig.savefig('MAR_Graph.png')

videoSteam.release()  
cv2.destroyAllWindows()