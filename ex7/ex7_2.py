from skimage import io
import matplotlib.pyplot as plt

pic = io.imread('data/bird_small.png') / 255
# io.imshow(pic)
# plt.show()
data = pic.reshape(128 * 128, 3)

from sklearn.cluster import KMeans

model = KMeans(n_clusters=16, n_init=100, n_jobs=1)
model.fit(data)
centroids=model.cluster_centers_
C=model.predict(data)
compressed_pic=centroids[C].reshape((128,128,3))
fig,ax=plt.subplots(1,2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()