import time

import cv2
import numpy as np
from ugot import ugot

#Connected camera
cap = cv2.VideoCapture(0)
got = ugot.UGOT()
# got.initialize('10.220.5.233')
got.initialize('10.220.5.228')
got.open_camera()

#Line position
linePos_1 = 380
linePos_2 = 430
lineColor_set = 0
base_speed = 30
enb = False

centerList = []

def nothing(x):
    pass

def capture_image(event, x, y, flags, param):
    global enb
    if event == cv2.EVENT_RBUTTONDOWN:
        enb=True
        # got.mecanum_motor_control(base_speed, base_speed, base_speed, base_speed)
        got.mecanum_move_xyz(0, base_speed, 0)
    if event == cv2.EVENT_MBUTTONDOWN:
        got.mecanum_stop()
        enb = False


cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 175, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 45, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    # ret, frame = cap.read()

    frame = got.read_camera_data()

    if frame is not None:
        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        tl = (100, 300)  # Điểm trên bên trái, di chuyển lên
        bl = (0, 400)  # Điểm dưới bên trái, di chuyển lên
        tr = (540, 300)  # Điểm trên bên phải, di chuyển lên
        br = (640, 400)

        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(data, matrix, (640, 480))

        find_line = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2GRAY)
        retval, find_line = cv2.threshold(find_line, 0, 255, cv2.THRESH_OTSU)
        colorPos_1 = find_line[linePos_1]
        colorPos_2 = find_line[linePos_2]

        try:
            lineColorCount_Pos1 = np.sum(colorPos_1 == lineColor_set)
            lineColorCount_Pos2 = np.sum(colorPos_2 == lineColor_set)

            lineIndex_Pos1 = np.where(colorPos_1 == lineColor_set)
            lineIndex_Pos2 = np.where(colorPos_2 == lineColor_set)

            if lineColorCount_Pos1 == 0:
                lineColorCount_Pos1 = 1
            if lineColorCount_Pos2 == 0:
                lineColorCount_Pos2 = 1

            leftPos_1 = lineIndex_Pos1[0][lineColorCount_Pos1 - 1]
            rightPos_1 = lineIndex_Pos1[0][0]
            centerPos_1 = int((leftPos_1 + rightPos_1)/2)

            leftPos_2 = lineIndex_Pos2[0][lineColorCount_Pos2 - 1]
            rightPos_2 = lineIndex_Pos2[0][0]
            centerPos_2 = int((leftPos_2 + rightPos_2)/2)

            center = int((centerPos_1 + centerPos_2)/2)

            centerList.append(center)

            cv2.line(transformed_frame, (leftPos_1, (linePos_1 + 30)), (leftPos_1, (linePos_1 - 30)), (255, 128, 64), 1)
            cv2.line(transformed_frame, (rightPos_1, (linePos_1 + 30)), (rightPos_1, (linePos_1 - 30)), (64, 128, 255), 1)
            cv2.line(transformed_frame, (0, linePos_1), (640, linePos_1), (255, 255, 64), 1)

            cv2.line(transformed_frame, (leftPos_2, (linePos_2 + 30)), (leftPos_2, (linePos_2 - 30)), (255, 128, 64), 1)
            cv2.line(transformed_frame, (rightPos_2, (linePos_2 + 30)), (rightPos_2, (linePos_2 - 30)), (64, 128, 255), 1)
            cv2.line(transformed_frame, (0, linePos_2), (640, linePos_2), (255, 255, 64), 1)

            cv2.line(transformed_frame, ((center - 20), int((linePos_1 + linePos_2) / 2)),
                     ((center + 20), int((linePos_1 + linePos_2) / 2)), (0, 0, 0), 1)
            cv2.line(transformed_frame, ((center), int((linePos_1 + linePos_2) / 2 + 20)),
                     ((center), int((linePos_1 + linePos_2) / 2 - 20)), (0, 0, 0), 1)
        except:
            center = None
            pass

        print(center)
        # enb = True
        # if enb is True:
        #     got.mecanum_move_xyz(0, base_speed,0)
        #     if center == None:
        #         got.mecanum_stop()
        #         enb = False
        #         for i in centerList:
        #             if centerList[-2] > 320:
        #                 got.mecanum_turn_speed_times(3,30,90,2)
        #                 got.mecanum_stop()
        #                 time.sleep(1)
        #                 enb = True
        #             elif centerList[-2] < 320:
        #                 got.mecanum_turn_speed_times(2,30,90,2)
        #                 got.mecanum_stop()
        #                 time.sleep(1)
        #                 enb =True
        #         got.mecanum_move_xyz(0,base_speed,0)

                # time.sleep(1)

        cv2.imshow('Frame', data)
        cv2.imshow('Frame Find Line', find_line)
        cv2.imshow('Frame Detected', transformed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            got.mecanum_stop()
            time.sleep(1)
            break

cap.release()
cv2.destroyAllWindows()