import cv2
import numpy as np
import dlib
from math import hypot
import time
import winsound

board = np.zeros((100, 500), np.uint8)
board[:] = 255
text = ""
text1 = []

# Load Camera
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Create virtual keyboard
keyboard = np.zeros((600, 800, 3), np.uint8)
keys = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
        7: "7", 8: "8", 9: "9", 10: "E", 11: "P"}
letter_index = 0
frames = 0
blinking_frame = 0
frames_to_blink = 10
frames_active_letter = 14
count = 0
password = ["1368", "5678", "1472", "2584", "4040", "5794"]
# a = 2

# Success Window
success = np.zeros((300, 800), np.uint8)
success[:] = 255
# Failure window
failure = np.zeros((300, 1000), np.uint8)
failure[:] = 255
error = np.zeros((200, 1000), np.uint8)
error[:] = 255


# Function to create Virtual keyboard
def letter(index, text, letter_light):
    # Keys
    x = 0
    y = 0
    if index == 0:
        x = 0
        y = 0
    elif index == 1:
        x = 200
        y = 0
    elif index == 2:
        x = 400
        y = 0
    elif index == 3:
        x = 600
        y = 0
    elif index == 4:
        x = 0
        y = 200
    elif index == 5:
        x = 200
        y = 200
    elif index == 6:
        x = 400
        y = 200
    elif index == 7:
        x = 600
        y = 200
    elif index == 8:
        x = 0
        y = 400
    elif index == 9:
        x = 200
        y = 400
    elif index == 10:
        x = 400
        y = 400
    elif index == 11:
        x = 600
        y = 400

    width = 200
    height = 200
    th = 3  # thickness

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    # to Highlight the numbers on the keyboard
    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)


# to find midpoint of the eye while calculating distance between two eyelids
def midpoint(p1, p2):
    return (p1.x + p2.x) // 2, (p1.y + p2.y) // 2,


# to get blinking ratio
def get_blinking_ratio(facial_landmark, eye_points):
    left_point = (facial_landmark.part(eye_points[0]).x, facial_landmark.part(eye_points[0]).y)
    right_point = (facial_landmark.part(eye_points[3]).x, facial_landmark.part(eye_points[3]).y)
    # hor_line = cv2.line(frame, left_point, right_point, (0, 0, 0), 1)

    top_point = midpoint(facial_landmark.part(eye_points[1]), facial_landmark.part(eye_points[2]))
    bottom_point = midpoint(facial_landmark.part(eye_points[5]), facial_landmark.part(eye_points[4]))
    # ver_line = cv2.line(frame, top_point, bottom_point, (0, 0, 0), 1)

    hor_line_len = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_len = hypot((top_point[0] - bottom_point[0]), (top_point[1] - bottom_point[1]))

    ratio = hor_line_len / (ver_line_len + 0.1)
    return ratio


# to get facial landmarks
def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    # for left eye
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    # for right eye
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye


while True:
    _, frame = cap.read()
    keyboard[:] = (0, 0, 0)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    rows, cols, _ = frame.shape
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)

    active_letter = keys[letter_index]

    for face in faces:
        landmark = predictor(gray, face)

        left_eye, right_eye = eyes_contour_points(landmark)

        left_eye_ratio = get_blinking_ratio(landmark, [36, 37, 38, 39, 40, 41])
        right_eye_ratio = get_blinking_ratio(landmark, [42, 43, 44, 45, 46, 47])
        avg_ratio = (left_eye_ratio + right_eye_ratio) / 2

        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)

        if avg_ratio > 6.5:
            # cv2.putText(frame, "Blinking", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 1)
            blinking_frame += 1
            frames -= 1

            # Show green eyes when closed
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

            if blinking_frame == frames_to_blink:
                winsound.PlaySound("sound.wav", winsound.SND_ASYNC)
                time.sleep(1)
                if active_letter != "E" and active_letter != "P":
                    count += 1
                    text1.append(active_letter)

                if active_letter == "P":
                    if len(text1) == 0:
                        error[:] = 255
                        cv2.putText(error, "Nothing to pop", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, 0, 5)
                        cv2.imshow("Error", error)

                    else:
                        board[:] = 255
                        count -= 1
                        text1.pop(count)

                if active_letter == "E":

                    if len(text1) != 4 :
                        print("Please enter right password")
                        error[:] = 255
                        cv2.putText(error, "Wrong password format", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2, 0, 5)
                        cv2.imshow("Error", error)

                    elif text.join(text1) in password:
                        print("Password matched")
                        cv2.putText(success, "Matched", (100, 200), cv2.FONT_HERSHEY_COMPLEX, 4, 0, 5)
                        cv2.imshow("Success", success)

                    else:
                        print("Password not matched")
                        cv2.putText(failure, "Not matched", (100, 200), cv2.FONT_HERSHEY_COMPLEX, 4, 0, 5)
                        cv2.imshow("Failure", failure)

        else:
            blinking_frame = 0
        break

    # Move cursor from one number to next number
    if frames == frames_active_letter:
        letter_index += 1
        frames = 0
    if letter_index == 12:
        letter_index = 0

    for i in range(12):
        if i == letter_index:
            light = True
        else:
            light = False
        letter(i, keys[i], light)

    cv2.putText(board, text.join(text1), (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, 0, 3)

    # Blinking loading bar
    percentage_blinking = blinking_frame / frames_to_blink
    loading_x = int(cols * percentage_blinking)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)

    cv2.imshow("keyboard", keyboard)
    cv2.imshow("Frame", frame)
    cv2.imshow("Board", board)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
