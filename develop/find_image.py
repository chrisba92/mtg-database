import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import imagehash
from PIL import Image, ImageFilter
import pickle

# =============================================================================
# Functions in script
# =============================================================================
def warp_image(card_contour):
    # create a min area rectangle from our contour
    _rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(_rect)
    box = np.int0(box)
    
    # create empty initialized rectangle
    rect = np.zeros((4, 2), dtype = "float32")
    
    # get top left and bottom right points
    s = box.sum(axis = 1)
    rect[0] = box[np.argmin(s)]
    rect[2] = box[np.argmax(s)]
    
    # get top right and bottom left points
    diff = np.diff(box, axis = 1)
    rect[1] = box[np.argmin(diff)]
    rect[3] = box[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    return warped

cwd = os.getcwd()

flist = glob.glob(cwd + '/test_images/*.jpg')

scale = 1

w = int(4000 * scale)
h = int(1824 * scale)

# Load the picture
frame = cv2.imread('test_images/warden_new.jpg')

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(frame,100,200)


# Convert to absolute grayscale
_,thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

imS = cv2.resize(thresh, (w, h))
plt.imshow(imS)
plt.show()

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sorted([ (cv2.contourArea(i), i) for i in contours ], key=lambda a:a[0], reverse=True)

# for _, contour in sorted_contours:
#     cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    

_, card_contour = sorted_contours[1]
#cv2.drawContours(frame, [card_contour], -1, (0, 255, 0), 2)

rect = cv2.minAreaRect(card_contour)
points = cv2.boxPoints(rect)
points = np.int0(points)

for point in points:
    cv2.circle(frame, tuple(point), 10, (0,255,0), -1)

imS = cv2.resize(frame, (w, h))
plt.imshow(imS)
plt.show()

warped = warp_image(card_contour)
plt.imshow(warped)
plt.show()

cv2.imwrite('warden.png', warped)

hash_ = imagehash.average_hash(Image.fromarray(warped))


# for file in flist:
#     # Load the picture
#     frame = cv2.imread(file)
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
#     # Convert to absolute grayscale
#     _,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
#     imS = cv2.resize(thresh, (w, h))
#     plt.imshow(imS)
#     plt.show()
    
#     contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     sorted_contours = sorted([ (cv2.contourArea(i), i) for i in contours ], key=lambda a:a[0], reverse=True)
    
#     # for _, contour in sorted_contours:
#     #     cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
    
#     _, card_contour = sorted_contours[1]
#     #cv2.drawContours(frame, [card_contour], -1, (0, 255, 0), 2)
    
#     rect = cv2.minAreaRect(card_contour)
#     points = cv2.boxPoints(rect)
#     points = np.int0(points)
    
#     for point in points:
#         cv2.circle(frame, tuple(point), 10, (0,255,0), -1)
    
#     imS = cv2.resize(frame, (w, h))
#     plt.imshow(imS)
#     plt.show()
    
#     warped = warp_image(card_contour)
#     plt.imshow(warped)
#     plt.show()
    
#     cv2.imwrite('test.png', warped)
    
#     hash_ = imagehash.average_hash(Image.fromarray(warped))
#     break