import argparse
import imutils
import trackbar
import cv2

global calibrationMode
blueRange = [(0, 100, 145), (255, 119, 173)]
yellowRange = [(0, 140, 80), (255, 150, 100)]
greenRange = [(0, 54, 93), (255, 117, 133)]
orangeRange = [(37, 155, 103), (59, 166, 124)]
redRange = [(0, 145, 114), (37, 157, 128)]
white = [(107, 122, 133), (128, 129, 141)]
colorRange = {'blue': [(0, 100, 145), (255, 119, 173), (255, 0, 0)],
              'yellow': [(0, 140, 80), (255, 150, 100), (0, 255, 255)],
              'green': [(0, 54, 93), (255, 117, 133), (0, 255, 0)],
              'orange': [(37, 155, 103), (59, 166, 124), (0, 128, 255)],
              'red': [(0, 145, 114), (37, 157, 128), (0, 0, 255)],
              'white': [(107, 122, 133), (128, 129, 141), (255, 255, 255)]}

def morphThresh(threshold):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    thresh = cv2.dilate(threshold, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.GaussianBlur(thresh, (1, 1), 0)
    return thresh


def trackFilteredObject(f, image, c):
    color = (0, 255, 0)
    validCnts = []
    thresh = f.shape[:2]

    if calibrationMode:
        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = trackbar.getTrackbarValues()
        thresh = cv2.inRange(f, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
        text = 'thresh'
    else:
        text = c
        color = colorRange[c][2]
        thresh = cv2.inRange(f, colorRange[c][0], colorRange[c][1])

    thresh = morphThresh(thresh)
    cv2.imshow("%s thresh" % text, thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    sortedCnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in sortedCnts:
        if cv2.contourArea(c) > 3000:
            validCnts.append(c)
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(image, [approx], -1, color, 2)
            if not calibrationMode:
                M = cv2.moments(c)
                cX = int(M['m10']/M['m00'])
                cY = int(M['m01']/M['m00'])
                center = (cX, cY)
                cv2.circle(image, center, 3, (255, 255, 0), 3)
                cv2.putText(image, text, (cX, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return validCnts


ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=False)
ap.add_argument("-v", "--video", required=False)
ap.add_argument("-m", "--mode", required=False, action='store_true')
args = vars(ap.parse_args())


if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

calibrationMode = False if not args.get("mode", False) else True
if calibrationMode:
    trackbar.setupTrackbars(['y', 'Cr', 'Cb'], "TrackBars")

while True:
    ret, image = camera.read()
    if not ret:
        break
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    if calibrationMode:
        trackFilteredObject(frame, image, None)
    else:
        trackFilteredObject(frame, image, 'blue')
        trackFilteredObject(frame, image, 'yellow')
        trackFilteredObject(frame, image, 'red')
        trackFilteredObject(frame, image, 'green')
        trackFilteredObject(frame, image, 'orange')
        trackFilteredObject(frame, image, 'white')

    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
