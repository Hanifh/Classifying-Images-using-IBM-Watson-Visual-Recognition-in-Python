import urllib.request
import cv2
import os
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

fish_image_url = ''
url.request.urlretrieve(fish_image_url, 'fish.jpg')
im = cv2.imread('fish.jpg')
im_corrected = cv2.cvtColor(im, cv2.COLOR.BGR2RGB)
plt.axis('off')
plt.imshow(im_corrected)
print("Original size of fish image is: {} Kilo Bytes".format(str(math.ceil((os.stat('fish.jpg').st_size)/1000))))


num_row = im.shape[0]
num_col = im.shape[1]
transform_image_for_KMeans = im.reshape(num_row*num_col, 3)

kmeans = KMeans(n_clusters = 8)
kmeans.fit(transform_image_for_KMeans)

cluster_centroids = np.asarray(kmeans.cluster_centers_, dtype= np.uint8)

labels = np.asarray(kmeans.labels_,dtype= np.uint8)
labels = labels.reshape(num_row, num_col)

compressed_image = np.ones((num_row, num_col, 3), dtype= np.uint8)

for r in range(num_row):
	for c in range(num_col):
		compressed_image[r,c,:] = cluster_centroids[labels[r,c], :]

cv2.imwrite("compressed_image.jpg", compressed_image)
compressed_image_im = cv2.imread("compressed_image.jpg")
compressed_image_im_corrected = cv2.cvtColor(compressed_image_im, cv2.COLOR.BGR2RGB)
plt.axis('off')
plt.imshow(compressed_image_im_corrected)