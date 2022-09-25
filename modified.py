# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 06:04:29 2021

@author: jain
"""

import cv2
import numpy as np


def pmf(frame): # Calculate Probability Mass Function (PMF)
   
    row = len(frame)
    col = len(frame[0])
    pixels = row * col # Total pixel count

    count = [0] * 256 # Store the count of each pixel value (0-255)

    for i in range(len(frame)):
        for j in range(len(frame[i])):

            count[frame[i][j]] += 1 

    PMF = np.array(count) 
    PMF = PMF / pixels # Store PMF percentage

    return PMF      

def cdf(PMF): # Calculate the Cumulative Distribution Function (CDF)

    CDF = [0]
    i =0
    temp = []
    
    for i in range(len(PMF)): # CDF for a given intensity is the PFM of that intensity + the sum of all the previous PMF vlaues.
    
        PMF[i]
        
        CDF.append(PMF[i] + sum(temp))

        temp.append(PMF[i])

    return CDF   

def equalize(channel, CDF): # Apply CDF to the color channel to equalize the image

    idx = 0
    height, length = channel.shape

    for i in range(height): # For each pixel, set its value to the CDF for its given intensity * 255
        for j in range(length):

            idx = channel[i][j]

            channel[i][j] = CDF[idx] * 255
    
    return channel

def gamma_corret(frame, gamma = 1): # Gamma Correction method 
   
    i_gamma = 1/gamma # invert gamma
    table = np.empty((1,256), np.uint8)
    for i in range(256):
        table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    corr_frame = cv2.LUT(frame, table)

    return corr_frame

# def color(frame): # Histogram Equalization on color channels

#     """
#     Break frame out into the 3 color channels
#     Calculate PMF
#     Calculate CDF
#     Multiply each pixel with its corrisponding CDF value for each channel
#     Combine the color channels
#     Display the frame
#     """

#     b, g, r = cv2.split(frame) # Separate frame into 3 color channels

#     # Calculate probability mass function
#     b_PMF = pmf(b)
#     g_PMF = pmf(g)
#     r_PMF = pmf(r)

#     # Calculate Cumulative Distribution Function (CDF)
#     b_CDF = cdf(b_PMF)
#     g_CDF = cdf(g_PMF)
#     r_CDF = cdf(r_PMF)

#     # Apply equilization
#     b_eq = equalize(b,b_CDF)
#     g_eq = equalize(g, g_CDF)
#     r_eq = equalize(r, r_CDF)  

#     # Combine channels
#     final = cv2.merge((b_eq, g_eq, r_eq))

#     # Gamma Correction
#     corr_frame = gamma_corret(final, gamma = 2)

#     cv2.imshow("Color Equilization", corr_frame) # Show the projection



def hist_equalize(image):
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = np.uint8(cdf/cdf.max()*255)
    equalized = cdf_normalized[image]
    return equalized

## Here k is the kernel size
def adaptive_Hist_Equalize(image, k = 4):
    image = image.copy()
    h, w = image.shape
    sh, sw = h//k, w//k
    for i in range(0, h, sh):
        for j in range(0, w, sw):
            image[i:i+sh, j:j+sw] = hist_equalize(image[i:i+sh, j:j+sw])
    return image

def color(frame): # Histogram Equalization on color channels

    """
    Break frame out into the 3 color channels
    Calculate PMF
    Calculate CDF
    Multiply each pixel with its corrisponding CDF value for each channel
    Combine the color channels
    Display the frame
    """

    b, g, r = cv2.split(frame) # Separate frame into 3 color channels


    # Apply equilization
    b_eq = adaptive_Hist_Equalize(b)
    g_eq = adaptive_Hist_Equalize(g)
    r_eq = adaptive_Hist_Equalize(r)  

    # Combine channels
    final = cv2.merge((b_eq, g_eq, r_eq))

    # Gamma Correction
    corr_frame = gamma_corret(final, gamma = 1)

    cv2.imshow("Color Equilization", corr_frame) # Show the projection

# Define Video out properties
cap = cv2.VideoCapture('nd.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('histogram_equalization.mp4', fourcc, 240, (720,480))

if __name__ == "__main__":

    while True:

        ret, frame = cap.read()

        if not ret: # If no frame returned, break the loop  
            break
        
        frame = cv2.resize(frame, (720,480))

        cv2.imshow("Video", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # denoise
        # blur = cv2.GaussianBlur(gray, (5,5), 0)
        # median = cv2.medianBlur(gray,10)
        # gray = median
        
        # Gamma Correction
        gamma_frame = gamma_corret(gray, gamma = .3)
        
        # denoise
        # blur = cv2.GaussianBlur(gamma_frame, (5,5), 0)
        # median = cv2.medianBlur(gamma_frame,5)
        # gray_frame = blur
        
        cv2.imshow("Gamma Corrected", gamma_frame) # Show the gamma correction on gray scale frame

        gray_PMF = pmf(gray)

        gray_CDF = cdf(gray_PMF)

        gray_eq = equalize(gray, gray_CDF) 
        gray_eq = adaptive_Hist_Equalize(gray,4)
        
        # filters
        # blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        # median = cv2.medianBlur(gray_eq, 5)
        # gray_eq = median
        
        cv2.imshow('Gray Equalization', gray_eq) # Show histogram equalization allpied on gray scale frame

        revert_frame = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR) # Convert gray scale back to color

        cv2.imshow('Gray to Color Converted Video', revert_frame) 

        color(frame) # Uncomment to show histogram equalization on color channels

        out.write(revert_frame) # Uncomment to save new output of desired frame
    
        # Condition to break the while loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

    # Wait for ESC key to be pressed before releasing capture method and closing windows
    cv2.waitKey(0)
    cap.release()
    out.release
    cv2.destroyAllWindows()
    
