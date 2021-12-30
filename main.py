import numpy as np
import cv2 as cv
import os
from win10toast import ToastNotifier

toast = ToastNotifier()
cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_lefteye_2splits.xml")

while True:
    eyes = ([])
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 5)
    ran = not np.any(faces)
    if ran:
        toast.show_toast("Notice", "Stay On Task And Pay Attention", duration=1)
        print('Stay On Task And Pay Attention')

    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 2, 5)
            ran = not np.any(eyes)
            if ran:
                toast.show_toast("Notice", "Stay On Task And Pay Attention", duration=1)
                print('Stay On Task And Pay Attention')
            else:
                for (ex, ey, ew, eh) in eyes:
                    cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
                print("Eyes Detected")


    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

    elif cv.waitKey(1) == ord('c'):
        os.system('cls')

cap.release()
cv.destroyAllWindows()
