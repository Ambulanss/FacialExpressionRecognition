#use python3 to avoid conflicts
import numpy as np
import cv2
import sys

def init(cascPath,hatPath):
    faceCascade = cv2.CascadeClassifier(cascPath) 
    hatImg = cv2.imread(hatPath, cv2.IMREAD_UNCHANGED) #read hat image with alpha channel
    video_capture = cv2.VideoCapture(0) #grab the camera 
    return faceCascade, hatImg, video_capture

def apply_smaller_image(frame, smallImg, x_shift = 0, y_shift = 0):
    alpha_s = smallImg[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        frame[ -smallImg.shape[0] + y_shift : y_shift, x_shift: smallImg.shape[1] + x_shift, c] = \
        (alpha_s * smallImg[:, :, c] + alpha_l * \
        frame[ -smallImg.shape[0] + y_shift: y_shift,\
        x_shift : smallImg.shape[1] + x_shift, c])
    return frame

def process_frame(cascade, smallImg, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
    faces, num_of_faces = cascade.detectMultiScale2(
        gray,
        scaleFactor=1.1,
        minNeighbors=9,
        minSize=(30, 30)
    )
    #print(faces)
    for (x1, y1, w1, h1) in faces:
        scaled = cv2.resize(smallImg, (0,0), fx=1.5 * w1/smallImg.shape[1],fy=1.5*w1/smallImg.shape[0])
        
        #calculate vertical and horizontal shifts according to face's height and width
        y_sh = 3* h1//8 + y1 
        x_sh = -w1//4 + x1
        try:
            frame = apply_smaller_image(frame, scaled, x_shift= x_sh, y_shift= y_sh)
        except ValueError:
            print("You are probably too close to the camera(your face is too big) or too close to the camera's upper fov boundary")
    return frame


#here goes the main logic
def main():
    faceCascade, hatImg, video_capture = init(          \
        "cascades/haarcascade_frontalface_default.xml", \
        "hat_big.png")
    #main loop
    while True:
        ret, frame = video_capture.read() #get 1 frame from webcam
        frame = process_frame(faceCascade, hatImg, frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #quit the loop when 'Q' is pressed
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()