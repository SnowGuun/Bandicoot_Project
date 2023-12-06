import cv2 
from cv2 import aruco
import numpy as np
import time
import math

def distanceCheck():
    # load in the calibration data
    calib_data_path = "calib_data/MultiMatrix.npz"

    calib_data = np.load(calib_data_path)
    print(calib_data.files)

    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]
    r_vectors = calib_data["rVector"]
    t_vectors = calib_data["tVector"]

    MARKER_SIZE = 2  # centimeters (measure your printed marker size)
    height = 0
    m1 = 10
    marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

    param_markers = aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)
    marker_display_info = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=param_markers
        )
        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, MARKER_SIZE, cam_mat, dist_coef
            )
            current_time = time.time()
            total_markers = range(0, marker_IDs.size)
            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                cv2.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                )
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                # Calculating the distance
                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                height = math.sqrt(distance * distance - m1*m1)

                if ids[0] not in marker_display_info or current_time - marker_display_info[ids[0]]['last_update_time'] >= 2:
                    marker_display_info[ids[0]] = {
                    'distance': round(distance, 2),
                    'height': round(height, 2),
                    'position': corners[0].ravel(),
                    'last_update_time': current_time
                    }
                # Draw the pose of the marker
                #point = cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                
                
            for marker_id, info in marker_display_info.items():
                
                cv2.putText(
                    frame,
                    f"ID: {marker_id} Height: {info['height']}",
                    info['position'],
                    cv2.FONT_HERSHEY_PLAIN,
                    1.3,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    if __name__ == "__main__":
        distanceCheck()