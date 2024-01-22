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

        self.d = 9
        self.p = 95 #testing height
        self.r = 20

        delta = math.pi * (5 / 4)
        theta_values = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        self.points = self.calculate_points(self.p, self.d, self.r, delta, theta_values)
        

    # Function to calculate points based on given parameters
    def calculate_points(self, p, d, r, delta, theta_values):
        points = []
        
        for theta in theta_values:
            phi = math.asin(r / p)  # Calculate phi for each theta
            dist = math.sqrt(p**2 + d**2 - 2*p*d*math.sin(phi)*math.cos(theta - delta))
            points.append(dist)
            
        return points
    
    def detect_markers(self, gray_frame):
        return aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)

    def draw_markers(self, frame, ids, corners):
        for id, corner in zip(ids, corners):
            if id[0] in [543, 109, 120]:
                cv2.polylines(frame, [corner.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)

    def print_point_distances(self):
        for i, dist in enumerate(self.points, start=1):
            print(f"Point {i} Distance: {dist:.2f} cm")
            

    # Main function for distance checking and height calculation
    def run(self):

        distances = {}
        last_update_time = 0
        updated_frquency = 1
        height_text = ""
        height_in_range = False
        user_at_point_msg = ""
        distances_printed = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_IDs, _ = self.detect_markers(gray_frame)  
            markers_detected = set()  # Set to store detected marker IDs
            
            if marker_corners:
                rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, self.MARKER_SIZE, self.cam_mat, self.dist_coef)
                for id, corner, tvec in zip(marker_IDs, marker_corners, tVec):
                    distances[id[0]] = np.linalg.norm(tvec[0])
                    markers_detected.add(id[0])
                self.draw_markers(frame, marker_IDs, marker_corners)       
            
            current_time = time.time()
            if 543 in markers_detected and 109 in markers_detected and current_time - last_update_time > updated_frquency:
               
                combined_distance = (distances[543] + distances[109]) / 2
                height = math.sqrt(combined_distance ** 2 - self.d ** 2)
                height_text = f"Current height:{round(height, 1)} cm"
                last_update_time = current_time
                height_in_range = 92 <= combined_distance <= 98
                if height_in_range:
                    height_text += " | In range"
                else:
                    height_text += " | Out of range"
            if 120 in markers_detected:
                user_distance_to_120 = distances[120]
                target_point_distance = self.points[0]  # Distance to Point 1
                tolerance = 3  # Tolerance for being at a point, 
                if abs(user_distance_to_120 - target_point_distance) <= tolerance and height_in_range:
                    #print("Point 1 is here")
                    user_at_point_msg = "You are at Point 1"
                else:
                    user_at_point_msg = ""

            if height_text:
                cv2.putText(frame, height_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            if user_at_point_msg:
                cv2.putText(frame, user_at_point_msg, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            
            

            #print the distances to the 8 points surrounding the chart
            if not distances_printed:         
                self.print_point_distances()
                    
                distances_printed = True
                

            #Display video frame
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        
        # Release the camera and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    pointTracker = pointTracker("calib_data/MultiMatrix.npz")
    pointTracker.run()
