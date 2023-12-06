import cv2
from cv2 import aruco
import numpy as np
def overviewHeight():
    cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read()

        
        if not ret:
             break
        


        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()



    return True

if __name__ == "__main__":
        overviewHeight()