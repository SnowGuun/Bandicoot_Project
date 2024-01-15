import cv2 
from cv2 import aruco
import numpy as np
import time
import math


class pointTracker():
    def __init__(self, calibration_path):
        self.calib_data_path = calibration_path
        self.calib_data = np.load(self.calib_data_path)
        self.cam_mat = self.calib_data["camMatrix"]
        self.dist_coef = self.calib_data["distCoef"]
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
        self.param_markers = aruco.DetectorParameters()
        self.MARKER_SIZE = 2  # centimeters
        self.cap = cv2.VideoCapture(0)
        self.m1 = 10 
        self.p = 95 #testing height
        self.r = 20 

    # Function to calculate points based on given parameters
    def calculate_points(self, p, d1, r, delta, theta_values):
        points = []
        for theta in theta_values:
            phi = math.asin(r / p)  # Calculate phi for each theta
            dist = math.sqrt(p**2 + d1**2 - 2*p*d1*math.sin(phi)*math.cos(theta - delta))
            points.append(dist)
        return points

    # Main function for distance checking and height calculation
    def eightPoint(self):

        distances = {}
        last_update_time = 0
        height_text = ""
        distances_printed = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_IDs, reject = aruco.detectMarkers(
                gray_frame, self.marker_dict, parameters=self.param_markers
            
            )
            #Marker detection
            markers_detected = set()  # Set to store detected marker IDs
            
            if marker_corners:
                rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                    marker_corners, self.MARKER_SIZE, self.cam_mat, self.dist_coef
                )

                # Calculate distance for each detected marker
                for ids, corners, i in zip(marker_IDs, marker_corners, range(marker_IDs.size)):
                    distance = np.sqrt(
                        tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                    )
                    
                    distances[ids[0]] = distance
                    markers_detected.add(ids[0])
                    if ids[0] in [543, 109]:
                        # Draw a yellow square around markers 543 and 109
                        cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)       
            
            # Calculate and display height every 2 seconds
            current_time = time.time()
            if 543 in markers_detected and 109 in markers_detected and current_time - last_update_time >= 2:
                combined_distance = (distances[543] + distances[109])/2
                height = math.sqrt(combined_distance ** 2 - self.m1 * 2)
                height_text = f"Current height:{round(height, 1)} cm"
                last_update_time = current_time
            elif current_time - last_update_time >= 2:
                height_text = ""

            #Show the text on screen
            if height_text:
                cv2.putText(
                    frame,
                    height_text,
                    (5, 30),  # Position of the text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            #print the distances to the 8 points surrounding the chart
            if not distances_printed:         
                delta = math.pi*7 / 6
                theta_values = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
            
                points = self.calculate_points(self.p, self.m1, self.r, delta, theta_values)
            
                for i, dist in enumerate(points, start=1):
                    print(f"Point {i} Distance: {dist:.2f} cm")
                    
                distances_printed = True
                

            #Display video frame
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        
        # Release the camera and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.eightPoint()
    

if __name__ == "__main__":
    pointTracker = pointTracker("calib_data/MultiMatrix.npz")
    pointTracker.run()
