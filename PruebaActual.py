import cv2
import numpy as np
import udF
import time
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(350000)

start_time = time.time()

    #-Obtención de la imagen de prueba

img = cv2.imread('/home/david/Escritorio/Texto Luz Blanca.jpg',0)
img = udF.imgRS(img,0.5) #Este resize está solo para hacer más rápidas las pruebas.

    #-Binarización de la Imagen

#img1 = udF.unNiblack(img)
#img1 = udF.cincoNiblack(img)
img1 = udF.wolf(img, 127, 0.2)
#udF.show_image(img1, "wolf")
print("Binarization done")
print("--- %s seconds ---" % (time.time() - start_time))

    #Busqueda de objetos y propiedades
        #Busqueda de regiones en la imagen

objMtx, nObj = udF.objSrch2(img1)
print("N de objetos encontrados =",nObj)
print("objTag done")
print("--- %s seconds ---" % (time.time() - start_time))

        #Boxing de OBJETOS

boxesLst = udF.boxing(objMtx, nObj)
boxesLst = udF.boxCleaning(boxesLst,img1)
print("Boxing done")
print("--- %s seconds ---" % (time.time() - start_time))

        #Formación de líneas de texto
groupedBoxes = udF.gouping_boxes(boxesLst)

            #transición a POO
cajas = udF.POOtransition(boxesLst)
#cajas = udF.POOtransition4groupedBoxes(groupedBoxes) #esto es para cuando se usa la función grouping_boxes
objMtx = udF.objMtxRW(objMtx,cajas)
objNum = len(cajas) #esto da el nuevo numero de objetos
print("POO transition done done")
print("--- %s seconds ---" % (time.time() - start_time))

######################################### Aclaración: ###############################################
# Cajas es donde se pondrán todas las características de las letras (es un object list)
#  objMatrix es una matriz del tamaño de la imagen original, donde se tienen el mapeo de los OBJETOS
#####################################################################################################

    #Efectos visuales
imgColored = udF.rgbObjColor(objMtx,objNum)
print("Coloring done")
print("--- %s seconds ---" % (time.time() - start_time))

imgSq = udF.DrawSq(imgColored,cajas)
print("Square drawing done")
print("--- %s seconds ---" % (time.time() - start_time))


cv2.imshow('5',imgColored)
cv2.waitKey(0)
