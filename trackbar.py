import cv2

def callback(value):
    pass

def setupTrackbars(filter, windowname):
    global range_filter
    global window_name
    range_filter = filter
    window_name = windowname
    cv2.namedWindow(window_name)
    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
        for j in filter:
            cv2.createTrackbar("%s_%s" % (i, j), window_name, v, 255, callback)

def getTrackbarValues():
    values = []
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (i, j), window_name)
            values.append(v)
    return values
