import cv2
import imutils
import numpy as np
import argparse

def ByImage(path):
    
    image = cv2.imread(path)
    image = imutils.resize(image, width = min(800, image.shape[1])) 
    detect(image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ByVideo(path):

    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path.')
        return

    print('Detecting people...')
    while video.isOpened():
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)
            
            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

def ByWebCam():   
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()

        frame = detect(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

def peopleDetection(args):
    image_path = args["image"]
    video_path = args['video']
    if str(args["camera"]) == 'true' : camera = True 
    else : camera = False

    if camera:
        print('[INFO] Opening Web Cam.')
        ByWebCam()
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        ByVideo(video_path)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        ByImage(image_path)

def detect(frame):
    coordinates, weights =  HOGopenCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    human = 1
    for x,y,w,h in coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'human {human}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        human += 1
    
    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total number of humans : {human-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)


    return frame    

def arguments():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument( "--video", default=None, help="path to Video File ")
    arg_parse.add_argument( "--image", default=None, help="path to Image File ")
    arg_parse.add_argument( "--camera", default=False, help="Set true if you want to use the camera.")
    args = vars(arg_parse.parse_args())

    return args

if __name__ == "__main__":
    HOGopenCV = cv2.HOGDescriptor()
    HOGopenCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = arguments()
    peopleDetection(args)