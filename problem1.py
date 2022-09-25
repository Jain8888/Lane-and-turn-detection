# # -*- coding: utf-8 -*-
# """
# Created on Tue Apr 13 06:04:29 2021

# @author: jain
# """
    
    
import cv2
import numpy as np
from matplotlib import pyplot as plt


# # convert video into frames
vidcap = cv2.VideoCapture('nd.mp4')
success,image = vidcap.read()
c = 0
while success:
  cv2.imwrite("frame%d.jpg" % c, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  c += 1

fps = c/24  # frames per second



for j in range(c):

    img = cv2.imread('frame'+str(j)+'.jpg')
    
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    
#     # plt.plot(cdf_normalized, color = 'b')
#     # plt.hist(img.flatten(),256,[0,256], color = 'r')
#     # plt.xlim([0,256])
#     # plt.legend(('cdf','histogram'), loc = 'upper left')
#     # plt.show()
    
    
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    img2 = cdf[img]
#     # hist,bins = np.histogram(img2.flatten(),256,[0,256])
    
#     # cdf = hist.cumsum()
#     # cdf_normalized = cdf * hist.max()/ cdf.max()
    
#     # plt.plot(cdf_normalized, color = 'b')
#     # plt.hist(img.flatten(),256,[0,256], color = 'r')
#     # plt.xlim([0,256])
#     # plt.legend(('cdf','histogram'), loc = 'upper left')
#     # plt.show()
    
    cv2.imwrite('img' + str(j) + '.jpg', img2)
    
    
    
    # make video of obtained frames
frame_array = []
for i in range(c):

    #reading each files
    img = cv2.imread('img'+str(i)+'.jpg')
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter('hist_eq.mp4',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
