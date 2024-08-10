import cv2
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize('10.220.5.233')
got.open_camera()


def nothing(x):
    pass


cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 175, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 45, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# History of points for smoothing
point_history = []

while True:
    frame = got.read_camera_data()

    if frame is not None:
        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Points for perspective transformation
        tl = (100, 400)
        bl = (0, 480)
        tr = (540, 400)
        br = (640, 480)

        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

        # Matrix to warp the image for bird's-eye view
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(data, matrix, (640, 480))

        # Convert to HSV and apply thresholding
        hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv_transformed_frame, lower, upper)

        # Apply a larger Gaussian blur to reduce noise
        blurred_mask = cv2.GaussianBlur(mask, (9, 9), 0)

        # Refine the threshold for black line detection
        threshold_value = 120  # Adjusted for better separation
        _, thresholded_image = cv2.threshold(blurred_mask, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Apply more robust morphological operations
        kernel = np.ones((9, 9), np.uint8)  # Larger kernel size
        cleaned_mask = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        edges = cv2.Canny(cleaned_mask, 50, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

        # Find contours in the mask
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_image = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

        # Draw the 10 points along the lane line and calculate their distance from the frame's center
        frame_center_x = result_image.shape[1] // 2  # x-coordinate of the frame's vertical center

        num_points = 5

        # Maintain history of x-coordinates for smoothing
        point_history = [[] for _ in range(num_points)]

        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Filter out small contours
                cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
                contour = contour.reshape(-1, 2)
                if len(contour) > 2:  # Need at least 3 points to fit a polynomial
                    # Fit a second-degree polynomial to the contour
                    fit = np.polyfit(contour[:, 1], contour[:, 0], 2)  # Fit y = a*x^2 + b*x + c

                    # Generate 10 evenly spaced y-values along the height of the image
                    y_values = np.linspace(0, result_image.shape[0] - 1, num_points).astype(int)

                    for idx, y in enumerate(y_values):
                        x = int(fit[0] * y ** 2 + fit[1] * y + fit[2])  # Calculate x value based on polynomial
                        point_history[idx].append(x)

                        # Smooth the x-coordinate by averaging the history
                        if len(point_history[idx]) > 10:  # Keep history length constant
                            point_history[idx].pop(0)

                        x_smoothed = int(np.mean(point_history[idx]))

                        cv2.circle(result_image, (x_smoothed, y), 2, (0, 0, 255), -1)

                        # Calculate and display the distance from the center
                        distance = x_smoothed - frame_center_x
                        cv2.putText(result_image, f"{distance}px", (x_smoothed + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0), 1, cv2.LINE_AA)
                        print(x_smoothed, y, distance)

        # Draw the 10 points along the vertical center of the frame
        y_values_center = np.linspace(0, result_image.shape[0] - 1, num_points).astype(int)

        for y in y_values_center:
            cv2.circle(result_image, (frame_center_x, y), 2, (255, 0, 0), -1)

        # Display results
        cv2.imshow("Original", data)
        cv2.imshow("Bird's Eye View", transformed_frame)
        cv2.imshow("Lane Detection - Contours", result_image)

        if cv2.waitKey(10) == ord('q'):
            break

cv2.destroyAllWindows()
