import cv2
import itertools
import numpy as np
import time
from imutils.video import WebcamVideoStream

def getTwoLargest(contours):
    '''
    This function returns the indices of two largest contours.
    :param contours: List of cv2 contour objects
    :return: Index of the contours with the two largest area
    '''
    x = cv2.contourArea(contours[0])
    y = cv2.contourArea(contours[1])

    largest = 0 if x > y else 1
    largest_area = x if x > y else y
    second = 1 if x > y else 0
    second_area = y if x > y else x

    for i in range(2, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > largest_area:
            # The old largest area is now the 2nd largest area
            second_area = largest_area
            # Set the largest area to the new area
            largest_area = area
            # Set 2nd largest index to old largest index
            second = largest
            # Set largest index to i
            largest = i
        elif area > second_area:
            # New area is in between old 2nd largest and largest area so new area becomes 2nd largest area.
            second_area = area
            second = i

    if second_area < 4000:
        # We need the area to be larger than some pre-defined value to count as a "hand". We arbitrarily chose 4000
        second = -1
    return largest, second


def getHands(contours):
    '''
    This function returns the contours of the two largest tuple objects which should correspond to the hands
    in the frame.
    :param contours: List of cv2 contour objects
    :return: A tuple containing the two largest tuple objects, if they exist. left
    is either None or the left contour object/hand. Right is always the right contour object/ahnd.
    '''
    if len(contours) < 2:
        return None, contours[0]
    # Grab the two largest contours
    largest, second = getTwoLargest(contours)
    if second < 0:
        # No left hand so return immediately
        return None, contours[largest]

    first_m = cv2.moments(contours[largest])
    first_x = int(first_m["m10"] / first_m["m00"])
    second_m = cv2.moments(contours[second])
    second_x = int(second_m["m10"] / second_m["m00"])
    left = contours[largest] if first_x < second_x else contours[second]
    right = contours[second] if first_x < second_x else contours[largest]
    return left, right


def getFingerTip(defects, contour, centroid, h):
    '''
    This function finds the farthest defect in the input contour and checks if it
    should be detected as a fingertip. If so, it returns True and the location of the defect/fingertip.
    Otherwise, is returns False and None.
    :param defects: list of defects of input contour
    :param contour: right hand contour
    :param centroid: centroid of right hand contour
    :return: -detected: boolean corresponding to whether fingertip is detected or not
             -farthest: location of farthest defect
    '''
    if defects is not None and centroid is not None:
        cx, cy = centroid

        # Get start points of all defects
        s = defects[:, 0][:, 0]

        # Get x and y coordinates of all defects
        sx = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        sy = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        # Calculate distance from centroid of contour to each defect
        x_dist = (cx - sx) ** 2
        y_dist = (cy - sy) ** 2
        dist = np.sqrt(y_dist + x_dist)

        # This one grabs all the indices of the defects below the centroid
        indices = np.nonzero((cy - sy) < 0)[0]

        # set all points found below the centroid to negative distance.
        dist[indices] = -1
        # Grab the defect with the largest distance that is above the centroid.
        highest_index = np.argmax(dist)

        # Calculate the ratio of the distance of the defect and centroid
        yratio = (h - sy[highest_index]) / (h - cy)

        if highest_index < len(s) and yratio > 2:
            # Get index of farthest defect
            farthest_s = s[highest_index]
            # Grab coordinate of fingertip
            farthest = tuple(contour[farthest_s][0])
            return True, farthest
        else:
            return False, None
    return False, None

#Code is based on https://becominghuman.ai/real-time-finger-detection
# -1e18fea0d1d4
def countDefects(contour, hull, defects, image):
    '''
    Returns the number of defects that are of interest which correspond to the number of gaps between fingers
    :param contour: right hand
    :param hull: convex hull of right hand
    :param defects: list of defects of right hand
    :param image: output frame
    :return: success: bool indicating whether there were any errors
             count: number of defects that are of interest
    '''
    count = 0
    # If 3 more less points make up the hull, it's probably not going to be a hand so disregard
    if len(hull) > 3:
        # Return if defects is Nonetype
        if type(defects) == type(None):
            return False, count
        # Get start point of each defect
        s = defects[:, 0][:, 0]
        # Get end point of each defect
        e = defects[:, 0][:, 1]
        # Get furthest point within the defect farthest from convex hull
        f = defects[:, 0][:, 2]

        # Get x,y coordinates of start points
        sx = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        sy = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        # Get x,y coordinates of end points
        ex = np.array(contour[e][:, 0][:, 0], dtype=np.float)
        ey = np.array(contour[e][:, 0][:, 1], dtype=np.float)

        # Get x,y coordinates of farthest points
        fx = np.array(contour[f][:, 0][:, 0], dtype=np.float)
        fy = np.array(contour[f][:, 0][:, 1], dtype=np.float)

        # Set a, b, c variables to use in cosine angle theorem
        a = np.sqrt((fx - sx) ** 2 + (fy - sy) ** 2)
        b = np.sqrt((fx - ex) ** 2 + (fy - ey) ** 2)
        c = np.sqrt(((sx - ex) ** 2 + (sy - ey) ** 2))

        # Calculate angles for all defects
        angles = np.arccos(np.divide((a ** 2 + b ** 2 - c ** 2), (2 * a * b)))
        count = np.count_nonzero(angles < np.pi / 2)

        # Get indices for defects that have angle less than 90 degrees. Most likely corresponds to a gap between fingers
        indices = np.nonzero(angles < np.pi / 2)[0]
        for i in indices:
            cv2.circle(image, tuple(contour[f][:, 0][i, :]), 8, (255, 0, 0), -1)
        return True, count
    return False, count


def getFingers(contour, frame, centroid, hull_indices, defects, h):
    '''
    Function for getting the fingers of the right hand.
    :param contour: Contour of the right hand
    :param frame: Output frame
    :param centroid: centroid of the right hand contour
    :param hull_indices: Indices of the convex hull of the contour
    :param defects: list containing the information about the defects of the contour
    :return: -count: the number of fingers held up
            - pointing: bool corresponding to whether 1 finger is pointing or not.
            - fingertip: location of finger point
    '''
    cx, cy = centroid
    bool, count = countDefects(contour, hull_indices, defects, frame)
    pointing, fingertip = False, None
    if count == 0:
        pointing, fingertip = getFingerTip(defects, contour, (cx, cy), h)
    if pointing or count > 0:
        # The number of fingers is 1 more than the total number of defects
        count += 1
    return count, pointing, fingertip


def inBounds(fingertip, rect):
    # Checks if fingertip is inside the rectangle defined by rect
    return fingertip[0] > rect[0][0] and fingertip[0] < rect[1][0] and fingertip[1] > rect[0][1] and fingertip[1] < \
           rect[1][1]


# Some global variables
cv_history = []
last_state = 0


def loop(h, cam, bg_model, background_set, rect, thickness, drawing, start, old_count):
    # Sleep to slow down the loop to allow processing time
    time.sleep(0.1)

    global cv_history, last_state, pipe
    frame = cam.read()
    # Blur to get rid of some noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.flip(frame, 1)
    # Variables used later for counting fingers
    restart_time = False
    count = -1

    if background_set:
        # Get foreground image
        mask = bg_model.apply(frame, learningRate=0)

        # Perform transformations on image to get binary image
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        img = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (41, 41), 0)
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

        # Find the hands
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            left, right = getHands(contours)

            # Moment and centroid of right hand
            M = cv2.moments(right)
            if M['m00'] == 0:
                return drawing, bg_model, background_set, True, count, restart_time, thickness
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)

            cx, cy = centroid
            # Convex hull of right hand
            hull_points = cv2.convexHull(right)
            hull_indices = cv2.convexHull(right, returnPoints=False)

            # Draw convex hull and centroid of right hand
            cv2.drawContours(frame, [right], -1, (255, 0, 0), 1)
            cv2.drawContours(frame, [hull_points], -1, (255, 0, 0), 1)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

            # Defects of right hand hull
            defects = cv2.convexityDefects(right, hull_indices)

            # Get fingers
            count, pointing, fingertip = getFingers(right, frame, centroid, hull_indices, defects, h)

            # Right hand pointing
            if pointing:
                # Left hand in image so we are drawing
                if left is not None and inBounds(fingertip, rect):
                    cv2.circle(frame, fingertip, 15, (0, 0, 255), -1)
                    drawing = True
                    pipe.send((fingertip, 11))
                    cv_history.append((fingertip, thickness))
                    last_state = 11
                # Left hand not in image so we are moving the cursor
                elif left is None and inBounds(fingertip, rect):
                    if drawing:
                        drawing = False
                        cv_history.append('start')
                    pipe.send((fingertip, 1))
                    last_state = 1
            # Not pointing
            else:
                # Need to put a token in history to start a new line in the cv image
                if drawing:
                    drawing = False
                    cv_history.append('start')
                # Right hand has 2-5 fingers so we're changing thickness
                if left is not None and count == 0 and last_state != 0:
                    pipe.send(((0, 0), 0))
                    last_state = 0
                    # last_state is used to prevent too many dup data
                # Changing thicnkess
                elif (count >= 2 and count < 6 and left is None) or (count == 0 and left is not None):
                    if count == old_count:
                        if time.time() - start > 2:
                            # Timer met, change thickness
                            thickness = count ** 2 if (count != 0 and count != 5) else thickness
                            pipe.send(((0, 0), count))
                            last_state = count
                            # cv_history.append(((0,0),thickness))
                    # New number of fingers held up. Retsrat timer
                    else:
                        restart_time = True
        else:
            pipe.send(((0, 0), 0))

        # Draw lines. First split the history on the "start" token to get each individual line.
        split = [list(y) for x, y in itertools.groupby(cv_history, lambda z: z == 'start') if not x]
        # Draw each line.
        for line in split:
            for i in range(len(line) - 1):
                cv2.line(frame, line[i][0], line[i + 1][0], (0, 0, 255), line[i][1])
    # If background not set, notify user to reset background
    else:
        cv2.putText(frame, "Press B to capture background. Do not include hands", (25, 25),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.75, (0, 255, 0), 2, cv2.LINE_4)
    cv2.rectangle(frame, rect[0], rect[1], (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)

    # Only set background if background not set or we reset
    if k == ord('b') and not background_set:
        bg_model = cv2.createBackgroundSubtractorMOG2()
        background_set = True

    # Reset background
    elif k == ord('r'):
        bg_model = None
        background_set = False

    # Clear the drawings on the cv image
    elif k == ord('c'):
        cv_history = []
        drawing = False

    # Quit
    elif k == ord('q'):
        cam.stop()
        cv2.destroyAllWindows()
        return drawing, bg_model, background_set, False, count, restart_time, thickness

    if (cv2.getWindowProperty('frame', 0) == -1):
        cam.stop()
        cv2.destroyAllWindows()
    return drawing, bg_model, background_set, True, count, restart_time, thickness


# Global variables for size of camera window
width = 640
height = 480
rect = [(40, 40), (600, 360)]


def main(pipe_object):
    cam = WebcamVideoStream(src=0).start()
    global pipe
    pipe = pipe_object

    background_set = False
    bg_model = None

    # Drawing variables
    thickness = 4
    drawing = False
    start = time.time()
    count = -1
    check = True
    # Main loop
    while check:
        drawing, bg_model, background_set, check, count, restart_time, thickness = loop(height, cam, bg_model,
                                                                                        background_set, rect, thickness,
                                                                                        drawing, start, count)
        if restart_time:
            start = time.time()


if __name__ == '__main__':
    main()
