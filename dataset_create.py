# import packages

import cv2            #opencv camera
import numpy          #numpy array
import sqlite3       #sqlite in databse

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');  # to detect face in camera
cam = cv2.VideoCapture(0);  # 0 is for web camera to detect faces


def insertorupdate(Id, Name, age):  # function is for sqlite
    conn = sqlite3.connect("sqlite.db ")  # connect database
    cmd = " SELECT * FROM STUDENTS WHERE ID=" + str(Id);
    cursor = conn.execute(cmd);
    isRecordExist = 0;  # assuming that there will be no record in our table

    for row in cursor:  # this for loop is for checking whether there are any record or exist in our students table
        isRecordExist = 1;  # if the record exists it wil be considered as 1
    if(isRecordExist==1):  #this wil help update or insert the values in our student table
        conn.execute("UPDATE STUDENTS SET Name=? WHERE ID=?",(Name, Id,))
        conn.execute("UPDATE STUDENTS SET age=? WHERE ID=?", (age, Id,))
    else:
        conn.execute("INSERT INTO STUDENTS(Id,Name,age) values(?,?,?)", (Id, Name, age))

    conn.commit()           # this will save all the changes I've than so far in the project
    conn.close()            # then we close the connection


#insert user defined values into tables

Id= input("Enter User Id: ")
Name= input("Enter User Name: ")
age= input("Enter User age: ")

insertorupdate(Id,Name,age)

#code for detecting faces

sampleNum=0;
while(True):
    ret,img=cam.read();
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                      #image convert into BGRGRAY COLOR
    faces= faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)     #scale face

    for(x,y,w,h) in faces:
        sampleNum= sampleNum+1;            #this will be incrementing if the faces are detected.this will increase the sample number
        cv2.imwrite("dataset/user."+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)   #x and y are position. if the faces are detected then I'm going  to
        # create a rectangle to show that this face is detected in my project
        cv2.waitKey(100)                          #waitkey is the time for showing the faces detected in our web camera that's 100sec
    cv2.imshow("face", img)                        #show  faces detected in the web camera
    cv2.waitKey(1);                                #waitkety for this is the delay type
    if(sampleNum>20):                              #if dataset is >20 break
        break;
cam.release()                                      #this will release the camera after tht we close it
cv2.destroyAllWindows()                               #ths will close or quit all the window
