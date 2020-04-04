import cv2
import numpy as np
import udF2
import time
import sys
sys.setrecursionlimit(350000)

start_time = time.time()

    #-Obtención de la imagen de prueba

img = cv2.imread('/home/david/Escritorio/ImagenesPrueba/Texto Luz Calida.jpg',0)
img = udF2.imgRS(img,0.5) #Este resize está solo para hacer más rápidas las pruebas.

    #-Binarización de la imagen

    #Lo seguiente se activa en caso de querer usar un solo niblack
winSize = 65
img1 = udF2.niblack(img, winSize, -2)
print("Niblack done")
print("--- %s seconds ---" % (time.time() - start_time))
img1 = udF2.invert(img1)
print("Invert done")
print("--- %s seconds ---" % (time.time() - start_time))

    #Uso de 5 niblacks
# img1 = udF2.cincoNiblack(img)
# print("Niblack done")
# print("--- %s seconds ---" % (time.time() - start_time))

    #-Quitar el padding que se puso en niblack, a fin de obtener una imagen del tamaño original
    #Solo aplcia si no se usan los 5 niblacks

img1 = udF2.getImgBk(img1, winSize)
print("getBack done")
print("--- %s seconds ---" % (time.time() - start_time))

    #checa vecindad con los 8 vecinos, pero solo son requeridos los de arriba y el centro izquierda. NO ENCUENTRA TRASLAPES DE OBJETOS!!!!!!!!!!!!!!!!!!!

objMtx, nObj = udF2.objSrch2(img1)
print("N de objetos encontrados =",nObj)
print("objTag done")
print("--- %s seconds ---" % (time.time() - start_time))

    #Boxing de OBJETOS

boxesLst = udF2.boxing(objMtx, nObj)
print("Boxing done")
print("--- %s seconds ---" % (time.time() - start_time))

    #colorea de un color aleatorio cada objeto distinto

imgColored = udF2.rgbObjColor(objMtx,nObj)
print("Coloring done")
print("--- %s seconds ---" % (time.time() - start_time))

    #Esto aun no jala
imgColored = udF2.DrawSq(imgColored,boxesLst)
print("Square drawing done")
print("--- %s seconds ---" % (time.time() - start_time))


cv2.imshow('5',imgColored)
cv2.waitKey(0)
