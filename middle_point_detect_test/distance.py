import cv2
import numpy as np
import concurrent.futures

def calculate_distance(focal_length, real_object_width, image_object_width):
    distance = (real_object_width * focal_length) / image_object_width
    return distance

def find_large_rectangle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_rectangles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1500:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                large_rectangles.append(approx)

    return large_rectangles

def draw_middle_point(frame, rectangle, focal_length, real_object_width):
    middle_point = np.mean(rectangle, axis=0, dtype=int)
    middle_point = tuple(middle_point.flatten())
    cv2.circle(frame, middle_point, 5, (0, 255, 0), -1)

    image_object_width = np.linalg.norm(rectangle[0] - rectangle[1])
    distance = calculate_distance(focal_length, real_object_width, image_object_width)

    if 50 <= distance <= 60:
        cv2.putText(frame, f"Distance: {distance:.2f}m - You can take a photo", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    elif distance < 50:
        cv2.putText(frame, f"Distance: {distance:.2f}m - Keep farther", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Distance: {distance:.2f}m - Keep closer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

def find_focal_length(calibration_image_path, object_width):
    calibration_image = cv2.imread(calibration_image_path)

    if calibration_image is None:
        print(f"Error: Unable to load image from {calibration_image_path}")
        return None

    gray = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        object_points = np.zeros((6 * 9, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        object_points *= object_width

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        _, camera_matrix, _, _, _ = cv2.calibrateCamera([object_points], [corners_subpix], gray.shape[::-1], None, None)

        focal_length = camera_matrix[0, 0]
        return focal_length

    else:
        print("Chessboard corners not found in the calibration image.")
        return None

def process_frame(frame, focal_length, real_object_width):
    large_rectangles = find_large_rectangle(frame)

    for rect in large_rectangles:
        frame = draw_middle_point(frame, rect, focal_length, real_object_width)

    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    frame_with_point = None
    real_object_width = 0.05

    calibration_image_path = 'calibration_image.jpg'
    calibration_object_width = 2.54
    calibration_focal_length = find_focal_length(calibration_image_path, calibration_object_width)
    focal_length = calibration_focal_length if calibration_focal_length is not None else 1000

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = process_frame(frame, focal_length, real_object_width)

        cv2.imshow('Large Rectangle with Middle Point', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
