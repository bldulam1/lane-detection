import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)


def displayLines(image, lines):
    lineImage = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImage


def regionOfInterest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage


def makeCoordinates(image, lineParameters):
    slope, intercept = lineParameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def averageSlopeIntercept(image, lines):
    leftFit = []
    rightFit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            leftFit.append((slope, intercept))
        else:
            rightFit.append((slope, intercept))
    leftFitAverage = np.average(leftFit, axis=0)
    rightFitAverage = np.average(rightFit, axis=0)
    leftLine = makeCoordinates(image, leftFitAverage)
    rightLine = makeCoordinates(image, rightFitAverage)
    return np.array([leftLine, rightLine])


def detectLanes(frame):
    cannyImage = canny(frame)
    croppedImage = regionOfInterest(cannyImage)
    lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100,
                            np.array([]), minLineLength=5, maxLineGap=5)
    averagedLines = averageSlopeIntercept(frame, lines)
    lineImage = displayLines(frame, averagedLines)
    comboImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)
    return comboImage


def analyzeVideo(video):
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        _, frame = cap.read()
        analyzedFrame = detectLanes(frame)

        cv2.imshow("result", analyzedFrame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


analyzeVideo('test2.mp4')
