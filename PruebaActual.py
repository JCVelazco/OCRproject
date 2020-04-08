import cv2
import numpy as np
import udF2
import time
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(350000)
class MyClass(object):
    def __init__(self, number,xmin,xmax,ymin,ymax):
        self.number = number
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

start_time = time.time()

    #-Obtención de la imagen de prueba

img = cv2.imread('/home/david/Escritorio/ImagenesPrueba/Texto Luz Calida.jpg',0)
img = udF2.imgRS(img,0.5) #Este resize está solo para hacer más rápidas las pruebas.

    #-Binarización de la Imagen

img1 = udF2.unNiblack(img)
#img1 = udF2.cincoNiblack(img)
print("Niblack done")
print("--- %s seconds ---" % (time.time() - start_time))

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

            #transición a POO
cajas = udF2.POOtransition(boxesLst)
objNum = len(cajas) #esto da el nuevo numero de objetos

######################################### Aclaración: ###############################################
# Cajas es donde se pondrán todas las características de las letras (es un object list)
#  objMatrix es una matriz del tamaño de la imagen original, donde se tienen el mapeo de los OBJETOS
#####################################################################################################

    #Efectos visuales
imgColored = udF2.rgbObjColor(objMtx,objNum)
print("Coloring done")
print("--- %s seconds ---" % (time.time() - start_time))

imgSq = udF2.DrawSq(imgColored,cajas)
print("Square drawing done")
print("--- %s seconds ---" % (time.time() - start_time))


cv2.imshow('5',imgColored)
cv2.waitKey(0)
