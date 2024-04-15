import os                                #read and write file
import cv2                               #opens camera
import numpy as np                       #array
from PIL import Image                   #image files that will read and write

recognizer =cv2.face.LBPHFaceRecognizer_create()                                #it recognizes the  faces in camera and trains the dataset
path = "dataset"                                                            #path for training the data set


def get_images_with_id(path):
    images_paths=[os.path.join(path,f) for f in os.listdir(path)]                      #opening images inside the dataset folder
    faces =[]            #using faces array that wil be the use of numpy
    ids=[]
    for single_image_path in images_paths:
        faceImg=Image.open(single_image_path).convert('L')       #L stands for luminance is nothing but incraesing the brithness in the images
        # line 16 is nothing but converting each image into black and white(gray) color
        faceNp=np.array(faceImg, np.uint8)                           #this line shows all the faces will be converted into array
        id=int(os.path.split(single_image_path)[-1].split(".")[1])
        print(id)
        faces.append(id)
        cv2.imshow("Training",faceNp)
        cv2.waitKey(10)
    return np.array(ids),faces
ids, faces=get_images_with_id(path)
recognizer.save("recognizer/trainingfaces.yml")             #the yml contains all the trained faces with their ID
cv2.destroyAllWindows()                         #quit

#I run thiS code like 50 this  wondering I made a mistake in my code but upon uninstalling and installing
# pip uninstall opencv-contrib-python # pip install opencv-contrib-python everything came back to normal