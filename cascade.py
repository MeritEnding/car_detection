import numpy as np
import cv2
import pafy

faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
noseCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_nose.xml')
url = 'https://www.youtube.com/watch?v=U5ET7VcE1q4'
video = pafy.new(url)
print('title=', video.title)

best = video.getbest(preftype='mp4')
print('best.resolution', best.resolution)

cap = cv2.VideoCapture(best.url)
while True:
    retval, frame = cap.read()
    if not retval:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        noses = noseCascade.detectMultiScale(roi_gray)
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(25)
    if key == 27:
        break
cv2.destroyAllWindows()
