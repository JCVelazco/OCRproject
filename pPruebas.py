import cv2
import numpy as np
import udF2
import time
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(350000)

start_time = time.time()

    #-Obtención de la imagen de prueba

img = cv2.imread('/home/david/Escritorio/ImagenesPrueba/Texto Luz Calida.jpg',0)
img = udF2.imgRS(img,0.5) #Este resize está solo para hacer más rápidas las pruebas.

    #-Binarización de la imagen

        #Un solo niblack
winSize = 65
img1 = udF2.niblack(img, winSize, -2)
print("Niblack done")
print("--- %s seconds ---" % (time.time() - start_time))
img1 = udF2.invert(img1)
print("Invert done")
print("--- %s seconds ---" % (time.time() - start_time))

img1 = udF2.getImgBk(img1, winSize)
print("getBack done")
print("--- %s seconds ---" % (time.time() - start_time))

        # Uso de 5 niblacks
# img1 = udF2.cincoNiblack(img)
# print("Niblack done")
# print("--- %s seconds ---" % (time.time() - start_time))


        #Operaciones morfológicas
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
# print("Morph done")
# print("--- %s seconds ---" % (time.time() - start_time))

    #Busqueda de objetos y propiedades
        #Busqueda de regiones en la imagen

objMtx, nObj = udF2.objSrch2(img1)
print("N de objetos encontrados =",nObj)
print("objTag done")
print("--- %s seconds ---" % (time.time() - start_time))

        #Boxing de OBJETOS

boxesLst = udF2.boxing(objMtx, nObj)
boxesLst = udF2.boxCleaning(boxesLst,img1)
print("Boxing done")
print("--- %s seconds ---" % (time.time() - start_time))

        #Medidas estadísticas
        
    #Efectos visuales
h,w = boxesLst.shape
imgColored = udF2.rgbObjColor(objMtx,h)
print("Coloring done")
print("--- %s seconds ---" % (time.time() - start_time))

imgSq = udF2.DrawSq(imgColored,boxesLst)
print("Square drawing done")
print("--- %s seconds ---" % (time.time() - start_time))


cv2.imshow('5',imgColored)
cv2.waitKey(0)
