import readexr_py
import numpy
import cv2


image=readexr_py.readexr_py('/mnt/d/workspace/Three-Filters-to-Normal/c_code/CmakeExample/0001.exr')


# 先做的简易版本，先用用看
a=image[0,:,:]
b=image[1,:,:]
g=image[2,:,:]
r=image[3,:,:]
z=image[4,:,:]


# because of the number may less than 0, it can not be show correctly.
# so we do this operation for show the image.

c = (b + 1) * 255 

cv2.imwrite('/mnt/c/Users/xxyy/Pictures/a.jpg',c)  # you can see the image 
cv2.imshow('example', c)


