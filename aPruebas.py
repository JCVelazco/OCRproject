#!/usr/bin/env python3

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print("removed")
from PIL import Image
import pytesseract
import argparse
import os
import cv2
import time
import numpy as np
sys.setrecursionlimit(120000000)

import utils

start_time = time.time()

#-Obtención de la imagen de prueba
img = cv2.imread('ImagenesProyecto/TextoRecto.jpg',0)
#img = cv2.imread('ImagenesProyecto/Texto Luz Blanca.jpg',0)
# img = cv2.imread('ImagenesProyecto/hola_como_estas.jpeg',0)
# img = cv2.imread('ImagenesProyecto/texto_prueba.jpg',0)
img = utils.imgRS(img,0.6) #Este resize está solo para hacer más rápidas las pruebas.



#Wolf thresholding
# window baja cuando la letra es clara y window alto cuando la letra es dificil de detectar
# k = 0 - 1 (0 -> menos erosion)
threshold_img = utils.wolf(img, 127, 0.2)
print("Wolf done")
print("--- %s seconds ---" % (time.time() - start_time))
utils.show_image(threshold_img, "wolf")

#Checa vecindad con los 8 vecinos, pero solo son requeridos los de arriba y el centro izquierda. No Overlapping
objMtx, nObj = utils.objSrch2(threshold_img)
print("N de objetos encontrados =",nObj)
print("objTag done")
print("--- %s seconds ---" % (time.time() - start_time))


#Colorea de un color aleatorio cada objeto distinto
imgColored = utils.rgbObjColor(objMtx,nObj)
print("Coloring done")
print("--- %s seconds ---" % (time.time() - start_time))
#utils.show_image(imgColored, "coloreada")

#Boxing de OBJETOS
boxesLst = utils.boxing2(objMtx, nObj, threshold_img)
print("Boxing done")
print("--- %s seconds ---" % (time.time() - start_time))

print("Wait")
print(boxesLst[0])

print(type(boxesLst[0]))
imgBoxes = np.copy(imgColored)
imgBoxes = utils.DrawSq(imgBoxes,boxesLst)
print("Square drawing done")
print("--- %s seconds ---" % (time.time() - start_time))
utils.show_image(imgBoxes, "boxes")

# to check by character is 0-2, to check word is 4-10, to check lines is 50+
#cluster by lines (50)
groupedBoxes = utils.grouping_boxes(boxesLst, imgColored, 40)
imgBoxes = utils.DrawSq(imgColored,groupedBoxes)
print("Cluster done")
print("--- %s seconds ---" % (time.time() - start_time))
utils.show_image(imgBoxes, "cluster")

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
# 	help="type of preprocessing to be done")
# args = vars(ap.parse_args())
#
# # load the example image and convert it to grayscale
# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("Image", gray)
#
# # check to see if we should apply thresholding to preprocess the
# # image
# if args["preprocess"] == "thresh":
# 	gray = cv2.threshold(gray, 0, 255,
# 		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
# # make a check to see if median blurring should be done to remove
# # noise
# elif args["preprocess"] == "blur":
# 	gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, threshold_img)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
# cv2.imshow("Image", image)
cv2.imshow("Output", threshold_img)
cv2.waitKey(0)
