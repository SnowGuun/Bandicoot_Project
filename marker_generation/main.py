import cv2
from cv2 import aruco
import os

if not os.path.exists("markers"):
    os.makedirs("markers")
# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# MARKER_ID = 0
MARKER_SIZE = 400  # pixels

# generating unique IDs using for loop
for id in range(20):  # genereting 20 markers
    # using funtion to draw a marker
    marker_image = aruco.generateImageMarker(marker_dict, id, MARKER_SIZE)
    cv2.imshow("img", marker_image)
    cv2.imwrite(f"markers/marker_{id}.png", marker_image)
    
    #cv2.waitKey(0)
    #break
    #meee
    #moussa