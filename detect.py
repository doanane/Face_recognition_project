import cv2
import os
import numpy as np
import sqlite3

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
recognizer =cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingfaces.yml")

def getprofile(id):
    conn=sqlite3.connect("sqlite.db")
    cursor=conn.execute("SELECT * FROM STUDENTS WHERE id?",(id,))
    profile=None

    for row in cursor:
        profile=row
    conn.close()
    return profile
# while(True):
#     ret,img=cam.read();
#     gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                      #image convert into BGRGRAY COLOR
#     faces=facedetect.detectMultiScale(gray,1.3,5)
#     for(x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#         id,conf=recognizer.predict(gray[y:y+h,x:x+w])    #using the trained yml file and predicting wht will be the values displayed in the output
#         profile=getprofile(id)
#         print(profile)
#         if(profile!=None):
#             cv2.putText(img,"Name:"+str(profile[1]),(x,y+h+20),cv2.FONT_HERSHEY_COMPLEX,1,(0,225,127),2)
#             cv2.putText(img,"Age:"+str(profile[1]),(x, y+ h + 45),cv2.FONT_HERSHEY_COMPLEX,1,(0,225,127),2)
#
#     cv2.imshow("FACE",img);
#     if(cv2.waitKey(1)==ord('q')):
#         break;
#
# cam.release()
# cv2.destroyAllWindows()


while True:
    try:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale frame
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face
            id, conf = recognizer.predict(gray[y:y+h, x:x+w])  # Predict the ID of the detected face
            profile = getprofile(id)  # Fetch profile from database based on ID
            print(profile)
            if profile is not None:
                cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 225, 127), 2)
                cv2.putText(img, "Age: " + str(profile[2]), (x, y+h+45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 225, 127), 2)

        cv2.imshow('FACE', img)  # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop when 'q' key is pressed
            break

    except KeyboardInterrupt:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()


#
# import cv2
# import os
# import numpy as np
# import sqlite3
#
# # Load the face detection classifier
# facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# # Initialize the camera
# cam = cv2.VideoCapture(0)
# # Load the LBPH face recognizer
# recognizer = cv2.face.LBPHFaceRecognizer_create()
#
# # Load the trained model
# recognizer.read("recognizer/trainingfaces.yml")
#
# # Function to fetch profile from SQLite database based on ID
# def getprofile(id):
#     conn = sqlite3.connect("sqlite.db")
#     cursor = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (id,))
#     profile = None
#     for row in cursor:
#         profile = row
#     conn.close()
#     return profile
#
# # Train the recognizer if necessary
# # You need to define your training data and labels here
# # faces_data = ...
# # labels = ...
# # recognizer.train(faces_data, np.array(labels))
#
# while True:
#     ret, img = cam.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale frame
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face
#         id, conf = recognizer.predict(gray[y:y+h, x:x+w])  # Predict the ID of the detected face
#         profile = getprofile(id)  # Fetch profile from database based on ID
#         print(profile)
#         if profile is not None:
#             cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 225, 127), 2)
#             cv2.putText(img, "Age: " + str(profile[2]), (x, y+h+45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 225, 127), 2)
#
#     cv2.imshow('FACE', img)  # Display the resulting frame
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop when 'q' key is pressed
#         break
#
# # Release the camera and close all windows
# cam.release()
# cv2.destroyAllWindows()
