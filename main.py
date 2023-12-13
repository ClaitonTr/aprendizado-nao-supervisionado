import numpy as np
import cv2 as cv

# carrega a imagem transforma em array 2d rgb
img = cv.imread('images/4.png')
Z = img.reshape((-1,3))

#converte em float
Z = np.float32(Z)

# conta o numero de cores únicas
na = np.array(img)
f = np.dot(na.astype(np.uint32),[1,256,65536]) 
nColours = len(np.unique(f))
print(nColours)

#aplicando o k-medias
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 500
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#salva a nova imagem
filename = 'result/k' + str(K) + '.png'
cv.imwrite(filename, res2)