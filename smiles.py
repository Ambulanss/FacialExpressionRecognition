#use python3 to avoid conflicts
import numpy as np
import cv2
import sys

#here goes the main logic
def main():
    
    cascPath = "cascades/haarcascade_frontalface_default.xml"
    smilePath = "cascades/smile.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    smileCascade = cv2.CascadeClassifier(smilePath)
    video_capture = cv2.VideoCapture(0) #grab the camera by the handler
    
    #main loop
    while True:
        ret, frame = video_capture.read() #read input from webcamera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces, num_of_faces = faceCascade.detectMultiScale2(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
           # flags=cv2.CV_HAAR_SCALE_IMAGE
        )
        print(faces)
        # Draw a rectangle around the faces
        for (x1, y1, w1, h1) in faces:
            print("x:", x1)
            face = gray[y1 : y1+h1, x1 : x1 + w1]            
            smiles = smileCascade.detectMultiScale(
                face,
                scaleFactor = 2.0,
                minNeighbors=4,
               ) 
            print(smiles)
            #draw a green rectangle around every face
            cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
            for (x, y, w, h) in smiles:
                #draw a yellow rectangle around every smile
                cv2.rectangle(frame, (x1+x, y1+y), (x1+x+w, y1+y+h), (0, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()