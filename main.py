import cv2
import numpy as np
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)
cap = cv2.VideoCapture("highway.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (980,540), cv2.INTER_AREA)
mask1 = np.zeros_like(frame)
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
lk_params = dict(winSize = (25, 25),
                maxLevel = 6,
                criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.3))
def select_point(event, x, y, flags , params):
   global point, point_selected, old_points
   if event== cv2.EVENT_LBUTTONDOWN:
       point = (x, y)
       #print(point)  To print the initial Points
       point_selected = True
       old_points= np.array([[x, y]], dtype=np.float32)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", select_point)
point_selected = False
point = ()
old_points = np.array([[]])
while True:
   ret, frame = cap.read()
   frame = cv2.resize(frame, (980, 540), cv2.INTER_AREA)
   gray_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
   roi = frame[300: 540, 400: 980]
   mask = object_detector.apply(roi)
   contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   for cnt in contours:
       area = cv2.contourArea(cnt)
       if area > 500:
           #cv2.drawContours(roi,[cnt], -1, (0,255,0), 1)
           #cv2.imshow("Roi",roi)
           x, y, w, h = cv2.boundingRect(cnt)
           cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
   if point_selected is True:
       cv2.circle(frame, point, 8, (0,0,255), 2)
       new_points, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
       old_gray= gray_frame.copy()
       x, y= new_points.ravel()
       x1, y1 = old_points.ravel()
       mask1 = cv2.line(mask1, (round(x), round(y)), (round(x1), round(y1)), (0,255,0), 2)
       cv2.imshow("mask1",mask1)
       old_points = new_points
       #print(x, y) To print the new points
       cv2.circle(frame, (round(x), round(y)), 4, (0, 255, 0), -1)
       cv2.rectangle(frame, (round(x-10), round(y-10)), (round(x+10), round(y+10)), (255, 0, 0), 1)
       cv2.putText(frame, "object", (round(x+10),round(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 0, 0),1)
       frame = cv2.add(frame, mask1)
   if (ret ==True):
       cv2.imshow("frame", frame)
       if cv2.waitKey(30) & 0xFF == ord("q"):
           break
   else:
       break

cap.release()
cv2.destroyAllWindows()

