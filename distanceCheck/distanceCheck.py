import cv2 
from cv2 import aruco
import numpy as np
import time
import math
def calculate_points(p, d1, r, delta, theta_values):
    points = []
    for theta in theta_values:
        phi = math.asin(r / p)  # Calculate phi for each theta
        dist = math.sqrt(p**2 + d1**2 - 2*p*d1*math.sin(phi)*math.cos(theta - delta))
        points.append(dist)
    return points

def distanceCheck():
    # load in the calibration data
    calib_data_path = "calib_data/MultiMatrix.npz"
    calib_data = np.load(calib_data_path)

    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]

    MARKER_SIZE = 2  # centimeters
    m1 = 10
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    param_markers = aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)
    distances = {}
    last_update_time = 0
    height_text = ""
    distances_printed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=param_markers
        
        )
        markers_detected = set()  # Set to store detected marker IDs
        
        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, MARKER_SIZE, cam_mat, dist_coef
            )

            for ids, corners, i in zip(marker_IDs, marker_corners, range(marker_IDs.size)):
                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                
                distances[ids[0]] = distance
                markers_detected.add(ids[0])
                if ids[0] in [543, 109]:
                    # Draw a yellow square around markers 543 and 109
                    cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)       
        
        current_time = time.time()
        if 543 in markers_detected and 109 in markers_detected and current_time - last_update_time >= 2:
            combined_distance = (distances[543] + distances[109])/2
            height = math.sqrt(combined_distance ** 2 - m1 * m1)
            height_text = f"Current height:{round(height, 1)} cm"
            last_update_time = current_time
        elif current_time - last_update_time >= 1:
            height_text = ""
        
        if height_text:
            cv2.putText(
                frame,
                height_text,
                (10, 50),  # Position of the text
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        if not distances_printed:
            p = 95  # Example height, replace with actual height calculation
            d1, r = 10, 20
            delta = math.pi*7 / 6
            theta_values = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        
            points = calculate_points(p, d1, r, delta, theta_values)
        
            for i, dist in enumerate(points, start=1):
                print(f"Point {i} Distance: {dist:.2f} cm")
                
            distances_printed = True
            

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    distanceCheck()
