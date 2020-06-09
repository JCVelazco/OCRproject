import cv2
import numpy as np
import udF
import time
import sys
sys.setrecursionlimit(120000000)

start_time = time.time()

#-Obtención de la imagen de prueba
#img = cv2.imread('ImagenesProyecto/test2.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('ImagenesProyecto/test1.jpg', cv2.IMREAD_GRAYSCALE)

img = udF.imgRS(img,0.6) #Este resize está solo para hacer más rápidas las pruebas.



#Wolf thresholding
# window baja cuando la letra es clara y window alto cuando la letra es dificil de detectar
# k = 0 - 1 (0 -> menos erosion)
threshold_img = udF.wolf(img, 127, 0.2)
print("Wolf done")
print("--- %s seconds ---" % (time.time() - start_time))
udF.show_image(threshold_img, "wolf")

#Checa vecindad con los 8 vecinos, pero solo son requeridos los de arriba y el centro izquierda. No Overlapping
objMtx, nObj = udF.objSrch2(threshold_img)
print("N de objetos encontrados =",nObj)
print("objTag done")
print("--- %s seconds ---" % (time.time() - start_time))


#Colorea de un color aleatorio cada objeto distinto
imgColored = udF.rgbObjColor(objMtx,nObj)
print("Coloring done")
print("--- %s seconds ---" % (time.time() - start_time))
#udF.show_image(imgColored, "coloreada")

#Boxing de OBJETOS
boxesLst = udF.boxing2(objMtx, nObj, threshold_img)
print("Boxing done")
print("--- %s seconds ---" % (time.time() - start_time))

print("Wait")
print(boxesLst[0])

print(type(boxesLst[0]))
imgBoxes = np.copy(imgColored)
imgBoxes = udF.DrawSq(imgBoxes,boxesLst)
print("Square drawing done")
print("--- %s seconds ---" % (time.time() - start_time))
udF.show_image(imgBoxes, "boxes")

# to check by character is 0-2, to check word is 4-10, to check lines is 50+
#cluster by lines (50)
groupedBoxes = udF.grouping_boxes(boxesLst, imgColored)
imgBoxes = udF.DrawSq(imgColored,groupedBoxes)
print("Cluster done")
print("--- %s seconds ---" % (time.time() - start_time))
udF.show_image(imgBoxes, "cluster")



print(f"I have {len(groupedBoxes)} groups of boxes (possible words)")
print('Prediction starts')
# the classifier was trained with images with text in black and background in white
udF.predict(groupedBoxes, udF.invert(threshold_img))
udF.show_image(imgBoxes, "cluster")
