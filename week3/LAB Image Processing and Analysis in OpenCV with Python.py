url = ''

import cv2
import urllib.request
from matplotlib import pyplot as plt
import numpy as np

urllib.request.urlretrieve('', 'image.jpg')

im = cv2.imread('image.jpg')
im_correct = cv2.cvtColor(im, cv2.COLOR.RGB2GRAY)

plt.hist(im_correct.ravel(), 256, [0, 256])
plt.title('Histogram of the Image')
plt.show()

color = ['b', 'g', 'r']

for i, col in enumerate(color):
	hist = cv2.calchist([im], [i], None, 256, [0,256])
	plt.plot(histr,color = col)
	plt.xlim([0, 256])

plt.show()
