import cv2
import os
import numpy as np

ppath = os.path.join(os.getcwd(), 'shapes_and_colors.png')
img = cv2.imread(ppath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", gray)
cv2.waitKey(0)

ret, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
height, width = binary.shape
bb = np.zeros((height, width), dtype=bool)
bb1 = np.where(binary == 0, False, True)
bb |= bb1
# bb2 = np.where(bb, 255, 0).astype(np.uint8)
bb2 = -bb.astype(np.uint8)
cv2.imshow("img", bb2)
cv2.waitKey(0)

img1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("img", img1)
cv2.waitKey(0)

img2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
area = [cv2.contourArea(cnt) for cnt in contours]
contours = list(filter(lambda cnt: cv2.contourArea(cnt) > 5, contours))
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
print(len(contours), area)

cv2.imshow("img", img)
cv2.waitKey(0)
