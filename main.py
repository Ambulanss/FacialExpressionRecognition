#use python3 to avoid conflicts
import numpy as np
import cv2
import sys

def init(cascPath,hatPath):
    faceCascade = cv2.CascadeClassifier(cascPath) 
    hatImg = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED) #read hat image with alpha channel
    video_capture = cv2.VideoCapture(0) #grab the camera 
    return faceCascade, hatImg, video_capture


#here goes the main logic
def main():
    faceCascade, hatImg, video_capture = init(\
        "cascades/haarcascade_frontalface_default.xml",\
        "hat.png")
    
    #main loop
    while True:
        ret, frame = video_capture.read() #read input from webcamera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        
        faces, num_of_faces = faceCascade.detectMultiScale2(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        print(faces)
        for (x1, y1, w1, h1) in faces:
            scaled = cv2.resize(hatImg, (0,0), fx=w1/75,fy=w1/75)
            alpha_s = scaled[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            y_shift = h1//2 - h1//8
            x_shift = -w1//6
            try:
                for c in range(0, 3):
                    frame[y1 - scaled.shape[0] + y_shift : y1 + y_shift, x1 + x_shift: x1 + scaled.shape[1] + x_shift, c] = \
                    (alpha_s * scaled[:, :, c] + alpha_l * \
                    frame[y1 - scaled.shape[0] + y_shift: y1 + y_shift,\
                    x1 + x_shift : x1 + scaled.shape[1] + x_shift, c])
            except ValueError:
                print("ValueError")
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()