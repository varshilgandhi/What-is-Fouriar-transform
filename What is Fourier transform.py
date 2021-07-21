# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:18:43 2021

@author: abc
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np

#Generate  a 2D sine wave image
x = np.arange(256)  # generate  1-D sine wave
y = np.sin(2* np.pi * x / 3)  #Control the frequency
y += max(y)  #offset wave by the max value to go out of negative range of sine

#create an array of that image and scaling it and rorate it you want
img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)
img = np.rot90(img)

#define fourier transform
#dft = discreate fourier transform
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

#shift this dft from corner to the center
dft_shift = np.fft.fftshift(dft)

#we need to calculate mangnitude spectrum because it has complex values
magnitude_sprectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))+ 1)


#plot image and magnitude sprectrum both 
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img)
ax1.title.set_text('Input image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_sprectrum)
ax2.title.set_text('FFT of image')
plt.show()

"""

NOTE : AS THE SPECIAL FREQUENCY INCREASES MEANS WHEN THE BAR GETS CLOSER THAN THE PICKS IN THE
       DFT GETTING FURTHER AWAY.

"""


#####################################################################################


#Let's perform above all things using real world image 

import cv2
from matplotlib import pyplot as plt
import numpy as np

#Read our image
img = cv2.imread("Alloy.jpg", 0)

#define fourier transform 
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

#shift dft from corner to center
dft_shift = np.fft.fftshift(dft)

#we need to calculate mangnitude spectrum because it has complex values
#define magnitude spretrum
magnitude_sprectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) +1)

#plot both image and magnitude sprectrum
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img)
ax1.title.set_text("Input image")
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_sprectrum)
ax2.title.set_text("FFT of image")
plt.show()


########################################

#perform above thing on different image

import cv2
from matplotlib import pyplot as plt
import numpy as np

#read our image
img = cv2.imread("monalisa.jpg", 0)

#define fourier transform
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

#shift dft from corner to center
dft_shift = np.fft.fftshift(dft)

#define magnitude sprectrum
magnitude_sprectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) +1)

#plot both image and magnitude sprectrum
fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img)
ax1.title.set_text("Input image")
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_sprectrum)
ax2.title.set_text("FFT of image")
plt.show()







