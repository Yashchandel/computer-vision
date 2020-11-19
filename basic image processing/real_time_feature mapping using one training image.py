import numpy as np
import matplotlib.pyplot as plt
import cv2

##training image getting keypoints and descriptors
image=cv2.imread(r'D:\[FreeCoursesOnline.Me] UDACITY - Natural Language Processing Nanodegree v1.0.0\img\yash_1.jpeg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

training_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

orb=cv2.ORB_create()
train_keypoint,train_descriptor=orb.detectAndCompute(training_gray,None)
keypoint_without_size=np.copy(image)
keypoint_with_size=np.copy(image)
cv2.drawKeypoints(image,train_keypoint,keypoint_without_size,color=(0,255,0))
cv2.drawKeypoints(image,train_keypoint,keypoint_with_size,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    testing_image = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    
    testing_image=cv2.cvtColor(testing_image,cv2.COLOR_BGR2RGB)

    testing_gray=cv2.cvtColor(testing_image,cv2.COLOR_RGB2GRAY)
    orb1=cv2.ORB_create()
    train_keypoint1,train_descriptor1=orb.detectAndCompute(testing_gray,None)
    keypoint_without_size1=np.copy(testing_image)
    keypoint_with_size1=np.copy(testing_image)
    cv2.drawKeypoints(testing_image,train_keypoint1,keypoint_without_size1,color=(0,255,0))
    cv2.drawKeypoints(testing_image,train_keypoint1,keypoint_with_size1,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches=bf.match(train_descriptor,train_descriptor1)
    matches=sorted(matches,key=lambda x:x.distance)
    result=cv2.drawMatches(image,train_keypoint,testing_gray,train_keypoint1,matches,testing_gray,flags=2)
    

    cv2.imshow('Input', result)


    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


