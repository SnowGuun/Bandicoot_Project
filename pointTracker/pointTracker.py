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
        self.height_readings = []
        self.reading_count = 5  # Number of readings to average
        self.range_buffer = 1.5  # Hysteresis buffer in cm
        self.dots = ['top', 'top-right', 'right', 'down-right', 'down', 'down-left', 'left', 'top-left']
        self.dots_to_points = {
            'top': 0, 'top-right': 1, 'right': 2, 'down-right': 3,
            'down': 4, 'down-left': 5, 'left': 6, 'top-left': 7}
        self.dots_sequence = ['top', 'top-right', 'right', 'down-right', 'down', 'down-left', 'left', 'top-left']
        self.current_dot_index = 0
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Adjust brightness, range [0-1]
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -3)  # Adjust exposure, value depends on camera
        cv2.namedWindow("Shimmer Scan - 8 Points", cv2.WINDOW_NORMAL)  # Create a named window
        cv2.setWindowProperty("Shimmer Scan - 8 Points", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set to fullscreen


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
            if id[0] in [543, 109]:
                cv2.polylines(frame, [corner.astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)

    def print_point_distances(self):
        for i, dist in enumerate(self.points, start=1):
            print(f"Point {i} Distance: {dist:.2f} cm")

    def guide_to_next_point(self, user_distance, height_in_range, key_press):
        tolerance = 0.7  # Tolerance for being at a point
        target_point_distance = self.points[self.current_point_index]

        if abs(user_distance - target_point_distance) <= tolerance and height_in_range:
            if key_press == ord('y'):
                self.current_point_index = (self.current_point_index + 1) % len(self.points)
            return f"You are at Point {self.current_point_index + 1} - Take Photo"
        return ""
    
    def calculate_middle_point(self, point1, point2):
        return (point1 + point2) / 2
    
    def calculate_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)
    
    def calculate_point_away_from_middle(self, middle_point, distance, direction):
        if direction == 'top':
            return middle_point - np.array([0, distance])
        elif direction == 'down':
            return middle_point + np.array([0, distance])
        elif direction == 'left':
            return middle_point - np.array([distance, 0])
        elif direction == 'right':
            return middle_point + np.array([distance, 0])
        elif direction == 'top-left':
            return middle_point - np.array([distance, distance])
        elif direction == 'top-right':
            return middle_point - np.array([-distance, distance])
        elif direction == 'down-left':
            return middle_point - np.array([distance, -distance])
        elif direction == 'down-right':
            return middle_point - np.array([-distance, -distance])
        else:
            return middle_point

    # Main function for distance checking and height calculation
    def run(self):

        distances = {}
        height_text = ""
        height_in_range = False
        move_to_dot_msg = ""
        user_at_point_msg = ""
        distances_printed = False
        self.current_point_index = 0
        dots = ['top', 'top-right', 'right', 'down-right', 'down', 'down-left', 'left', 'top-left']
        current_dot = 0
        dots_to_points = {
            'top': 0, 'top-right': 1, 'right': 2, 'down-right': 3,
            'down': 4, 'down-left': 5, 'left': 6, 'top-left': 7
        }
        

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
                    distance = np.linalg.norm(tvec[0])
                    distances[id[0]] = distance
                    markers_detected.add(id[0])
        
                self.draw_markers(frame, marker_IDs, marker_corners)

                if marker_IDs is not None:
                        marker_IDs_list = [id[0] for id in marker_IDs]
                        if 543 in marker_IDs_list and 109 in marker_IDs_list:
                            marker543_index = marker_IDs_list.index(543)
                            marker109_index = marker_IDs_list.index(109)
                            marker543_center = np.mean(marker_corners[marker543_index][0], axis=0)
                            marker109_center = np.mean(marker_corners[marker109_index][0], axis=0)
                            middle_point = self.calculate_middle_point(marker543_center, marker109_center)

                            frame = cv2.circle(frame, tuple(middle_point.astype(int)), 5, (0, 0, 255), -1)
                            current_direction = dots[current_dot]
                            distance_from_middle = 200  # Specify the distance in centimeters
                            dot_point = self.calculate_point_away_from_middle(middle_point, distance_from_middle, current_direction)
                            frame = cv2.circle(frame, tuple(dot_point.astype(int)), 5, (255, 0, 0), -1)
                            combined_distance = (distances[543] + distances[109]) / 2
                            current_height = math.sqrt(combined_distance ** 2 - self.d ** 2)
                            self.height_readings.append(current_height)

                            if len(self.height_readings) > self.reading_count:
                                self.height_readings.pop(0)

                            average_height = sum(self.height_readings) / len(self.height_readings)
                            height_text = f"Current height: {average_height:.1f} cm"
                            lower_bound = 93.5 - self.range_buffer
                            upper_bound = 98.5 + self.range_buffer
                            height_in_range = lower_bound <= average_height <= upper_bound
                            if height_in_range:
                                height_text += " | In range"
                                
                            else:
                                height_text += " | Move near 95cm height"

                            # Display the current dot direction
                            frame = cv2.putText(frame, f"Dot Direction: {current_direction}", (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        if 120 in marker_IDs_list:
                            distance_to_120 = distances[120]
                            distance_text = f"Current camera position to chart: {distance_to_120:.2f} cm "
                            cv2.putText(frame, distance_text, (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                            index_of_120 = marker_IDs_list.index(120)
                            cv2.polylines(frame, [marker_corners[index_of_120].astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
                        if 120 in marker_IDs_list and height_in_range:
                            user_distance_to_120 = distances[120]
                            key = cv2.waitKey(1) & 0xFF
                            user_at_point_msg= self.guide_to_next_point(user_distance_to_120, height_in_range, key)
                                                 

            if height_text:
                cv2.putText(frame, height_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            if move_to_dot_msg:
                cv2.putText(frame, move_to_dot_msg, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            if user_at_point_msg:
                cv2.putText(frame, user_at_point_msg, (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            
            #print the distances to the 8 points surrounding the chart
            if not distances_printed:         
                self.print_point_distances()
                    
                distances_printed = True
                

            #Display video frame
            cv2.imshow("Shimmer Scan - 8 Points", frame)
            key = cv2.waitKey(1)
            if key == ord('y') or key == ord('Y'):
                self.current_point_index = (self.current_point_index +1 ) % len(self.points)
                current_dot = self.current_point_index
                move_to_dot_msg = f"Move to Point {self.current_point_index + 1}"


            # Exit the program on 'q' key press
            elif key == ord('q') or key == 27:  # 27 is the ASCII code for the Escape key
                break
        
        # Release the camera and close all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    pointTracker = pointTracker("calib_data/MultiMatrix.npz")
    pointTracker.run()
