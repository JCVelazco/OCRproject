import numpy as np
import cv2
import random
import time
import math
from statistics import mode 
    ######################Binarizacion de imagen#####################

def avgF(img, ventana):
    new_image = cv2.blur(img,(ventana, ventana))
    return new_image

def imgRS(img, factor):
    scale_percent = factor # percent of original size
    h,w= img.shape
    width = int(w * scale_percent )
    height = int(h * scale_percent )
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def wolf(img, blockSize, k):
    result = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY_INV, blockSize, k, binarizationMethod=cv2.ximgproc.BINARIZATION_WOLF)
    return result

def niblack(img, ventana, k):
    rad = int((ventana-1)/2)
    img = np.pad(img,rad,'constant', constant_values=(0,0))
    h,w= img.shape
    imgBas = np.zeros((h, w))
    resultado = np.zeros((h, w))

    mean = avgF(img,ventana)
    mean2 = np.multiply(mean,mean)
    imgBas = np.multiply(img,img)
    meanSquare = avgF(imgBas,ventana)
    imgBas = np.subtract(meanSquare, mean2)
    deviation = np.sqrt(imgBas)

    for i in range(h):
        for j in range(w):
            Comparison = mean[i,j] + k*deviation[i,j]
            if img[i][j] > Comparison:
                resultado[i,j] = 255
            else:
                resultado[i,j] = 0

    return resultado


def invert(img):
    h,w= img.shape
    resultado = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if img[i][j] > 128:
                resultado[i,j] = 0
            else:
                resultado[i,j] = 255


    return resultado

def invert_v2(img):
    img = (255-img)
    return img


def getImgBk(img,ventana):
    h,w= img.shape
    h2 = h+1-ventana
    w2 = w+1-ventana
    rad = int((ventana-1)/2)
    imgA = np.zeros((h2,w2))

    for i in range(h):
        for j in range(w):
            if ((i > rad-1 and j > rad-1 and i < h-rad and j < w-rad) and img[i,j] > 0):
                imgA[i-rad][j-rad] = img[i,j]

    return imgA

def cincoNiblack(img):

    winSize = 5
    img1 = niblack(img, winSize, -2)
    img1 = invert(img1)
    img1 = getImgBk(img1, winSize)

    winSize = 17
    img2 = niblack(img, winSize, -2)
    img2 = invert(img2)
    img2 = getImgBk(img2, winSize)

    winSize = 37
    img3 = niblack(img, winSize, -2)
    img3 = invert(img3)
    img3 = getImgBk(img3, winSize)

    winSize = 65
    img4 = niblack(img, winSize, -2)
    img4 = invert(img4)
    img4 = getImgBk(img4, winSize)

    winSize = 257
    img5 = niblack(img, winSize, -2)# start_time = time.time()
    img5 = invert(img5)
    img5 = getImgBk(img5,winSize)

    imgF = imgSynth(img1,img2,img3,img4,img5)

    return imgF

def imgSynth(img1,img2,img3,img4,img5):
    h,w= img5.shape
    imgF = np.zeros((h, w))

    count = 0
    for i in range(h):
        for j in range(w):
            count = 0
            if img1[i,j] > 128: count = count + 1
            if img2[i,j] > 128: count = count + 1
            if img3[i,j] > 128: count = count + 1
            if img4[i,j] > 128: count = count + 1
            if img5[i,j] > 128: count = count + 1

            if count > 2:
                imgF[i,j] = 255
            else:
                imgF[i,j] = 0

    return imgF

######################Busqueda de objetos y caracteristicas#####################

def objSrch2(img):
    img = np.pad(img,1,'constant', constant_values=(0,0)) #Padding
    h,w = img.shape
    objectCount = 1
    depthCounter = 0
    limit = 3000
    objMtx = np.zeros((h,w)) #Matriz para mapeado de objetos
    for i in range(h):
        for j in range(w):
            if img[i,j] != 0 and objMtx[i,j] == 0:
                depthCounter = 0
                objMtx, depthCounter = objectFinder(img,objectCount,i,j,objMtx,depthCounter,limit)
                objectCount = objectCount + 1

    return objMtx, objectCount

def objectFinder(img,tag,i,j,objFnd,depthCounter,limit):
    h,w = img.shape
    depthCounter = depthCounter + 1
    objFnd[i,j] = tag
    if(img[i+1,j] != 0 and objFnd[i+1,j] == 0) and depthCounter < limit: #
        objFnd,depthCounter = objectFinder(img,tag,i+1,j,objFnd,depthCounter,limit)
    if(img[i+1,j-1] != 0 and objFnd[i+1,j-1] == 0) and depthCounter < limit: #
        objFnd,depthCounter = objectFinder(img,tag,i+1,j-1,objFnd,depthCounter,limit)
    if(img[i+1,j+1] != 0 and objFnd[i+1,j+1] == 0) and depthCounter < limit:#
        objFnd,depthCounter = objectFinder(img,tag,i+1,j+1,objFnd,depthCounter,limit)
    if(img[i,j-1] != 0 and objFnd[i,j-1] == 0) and depthCounter < limit: #
        objFnd,depthCounter = objectFinder(img,tag,i,j-1,objFnd,depthCounter,limit)
    if(img[i,j+1] != 0 and objFnd[i,j+1] == 0) and depthCounter < limit: #
        objFnd,depthCounter = objectFinder(img,tag,i,j+1,objFnd,depthCounter,limit)
    if(img[i-1,j+1] != 0 and objFnd[i-1,j+1] == 0) and depthCounter < limit:
        objFnd,depthCounter = objectFinder(img,tag,i-1,j+1,objFnd,depthCounter,limit)
    if(img[i-1,j] != 0 and objFnd[i-1,j] == 0) and depthCounter < limit:
        objFnd,depthCounter = objectFinder(img,tag,i-1,j,objFnd,depthCounter,limit)
    if(img[i-1,j-1] != 0 and objFnd[i-1,j-1] == 0) and depthCounter < limit:
        objFnd,depthCounter = objectFinder(img,tag,i-1,j-1,objFnd,depthCounter,limit)
    depthCounter = depthCounter - 1

    return objFnd, depthCounter

def objTag(img):
    img = np.pad(img,1,'constant', constant_values=(0,0)) #Padding
    h,w = img.shape
    objMtx = np.zeros((h,w)) #Matriz para mapeado de objetos

    objCount = 1 #Cuenta para llevar la asignación de numero de objeto

    for i in range(h):
        for j in range(w):
            if(img[i,j] > 0): #PIXEL ACTIVO
            #Checar los vecinos. El primer vecino encontrado será del que agarre el tag de objeto.
                if(objMtx[i-1,j] != 0):
                    objMtx[i,j] = objMtx[i-1,j]
                elif(objMtx[i-1,j-1] != 0):
                    objMtx[i,j] = objMtx[i-1,j-1]
                elif(objMtx[i-1,j+1] != 0):
                    objMtx[i,j] = objMtx[i-1,j+1]
                elif(objMtx[i,j-1] != 0):
                    objMtx[i,j] = objMtx[i,j-1]
                else:
                    #Si no tiene vecinos nombrados, se le dará un nuevo tag de objeto
                    objMtx[i,j] = objCount
                    objCount = objCount + 1
    objMtx = getImgBk(objMtx,3)

    return objMtx, objCount-1

def boxing(objMtx,objNums):
    h,w = objMtx.shape
    objsBxd = np.zeros((objNums + 1,4)) # Filas arriba, Fila abajo, Columna izq, Columna derecha en orden de las columnas 0 -> 3
    for i in range(h):
        for j in range(w):
            if objMtx[i,j] != 0:
                index = int(objMtx[i,j])
                if objsBxd[index,0] == 0:
                    objsBxd[index,0] = i #posicion para i mas chica
                    objsBxd[index,2] = j#posicion para j mas chica
                else:
                    if i > objsBxd[index,1]:
                        objsBxd[index,1] = i #Posicion para i mas grande
                    if j > objsBxd[index,3]:
                        objsBxd[index,3] = j #posicion para j mas grande
                    elif j < objsBxd[index,2]:
                        objsBxd[index,2] = j#posicion para j mas chica


    return objsBxd

def boxCleaning(boxesLst,img):
    h,w = img.shape
    a,b = boxesLst.shape
    realBoxes = np.zeros((a,b))
    cE = 0 #contador extra
    for i in range(a):
        if(boxesLst[i,2] < w and boxesLst[i,3] < w and boxesLst[i,0] < h and boxesLst[i,1] < h ): #Que no se pase arriba
            if(boxesLst[i,2] >0 and boxesLst[i,3] >0 and boxesLst[i,0] >0 and boxesLst[i,1] >0 ): #que no se pase abajo
                realBoxes[cE,0] = boxesLst[i,0]
                realBoxes[cE,1] = boxesLst[i,1]
                realBoxes[cE,2] = boxesLst[i,2]
                realBoxes[cE,3] = boxesLst[i,3]
                cE = cE + 1

    #delete all boxes where all its values are the same
    realBoxes = [element for element in realBoxes if element[0] != element[1] or element[2] != element[3]]
    return realBoxes


def remove_noicy_boxes(list_boxes, width_check, heigh_check):

    # delete all the ones that are less of the medium high/width
    if width_check == True:
        #remove large width objects
        widthSizes = [abs(element[3] - element[2]) for element in list_boxes] 
        medianX = np.mean(widthSizes)
        #desvEstandartX = math.sqrt(np.std(widthSizes))
        list_boxes = [bounding_box for bounding_box in list_boxes if  abs(bounding_box[3] - bounding_box[2]) < medianX*20]

    if heigh_check == True:
        #remove big and small heigh objects
        heighSizes = [abs(element[1] - element[0]) for element in list_boxes] 
        medianY = np.mean(heighSizes)
        #desvEstandartY = math.sqrt(np.std(heighSizes))

        list_boxes = [bounding_box for bounding_box in list_boxes if abs(bounding_box[1] - bounding_box[0]) > medianY/2 and abs(bounding_box[1] - bounding_box[0]) < medianY*2]


    
    return list_boxes



def grouping_boxes(list_boxes, img, x_maxgap):
    h, w, channels = img.shape

    list_boxes = remove_noicy_boxes(list_boxes, width_check=True, heigh_check=False)
    
    # sort by mid point between ymin 
    sorted_boxes = sorted(list_boxes, key=lambda element: ((element[0]+element[1])/2)) 

    # get distance between lines
    y_gap_distance = list((((sorted_boxes[index+1][0]+sorted_boxes[index+1][1])/2) - ((sorted_boxes[index][0]+sorted_boxes[index][1])/2)) for index in range(len(sorted_boxes) - 1))
    gap_dictionary = {element: y_gap_distance.count(element) for element in set(y_gap_distance)}
    most_repetead_element = mode(y_gap_distance)
    y_gap_distance = [value for value in y_gap_distance if value != most_repetead_element]

    print(mode(y_gap_distance))
    print(np.median(y_gap_distance))
    print(np.mean(y_gap_distance))
    print(np.std(y_gap_distance))
    print(np.sqrt(np.std(y_gap_distance)))
    #distance varitions between boxes (in y)
    #y_maxgap = 3
    y_maxgap = np.median(y_gap_distance) + np.sqrt(np.std(y_gap_distance))
    groups = [[sorted_boxes[0]]]
    for element in sorted_boxes[1:]:
        if abs(((element[0]+element[1])/2) - ((groups[-1][-1][0]+groups[-1][-1][1])/2)) <= y_maxgap:
            groups[-1].append(element)
        else:
            groups.append([element])

    final_groups = []

    

    for group_section in groups:
        # sort the groups by X position
        xgroup_section = sorted(group_section, key=lambda element: (element[2])) 
        final_groups.append([xgroup_section[0]])

        for element in xgroup_section[1:]:
            if abs((element[2]) - (final_groups[-1][-1][3])) <= x_maxgap:
                final_groups[-1].append(element)
            else:
                final_groups.append([element])
    

    # una vez teniendo los gropus junto todos sus boxes: 
    new_list_boxes = []
    for group_items in final_groups:
        ymin = h
        ymax = 0
        xmin = w
        xmax = 0

        for box in group_items:
            if box[0] < ymin:
                ymin = box[0]
            
            if box[1] > ymax:
                ymax = box[1]

            if box[2] < xmin:
                xmin = box[2]

            if box[3] > xmax:
                xmax = box[3] 
        
        new_list_boxes.append([ymin, ymax, xmin, xmax])

    remove_noicy_boxes(new_list_boxes, False, True)
    return new_list_boxes

######################Efectos visuales#####################

def rgbObjColor(objMtx,objCount):
    h,w = objMtx.shape
    imgF = np.zeros((h,w,3), np.uint8)
    saturation = np.zeros((objCount + 1,3))
    for i in range(objCount + 1):
        if(i > 0):
            saturation[i,0] = random.randint(0,253)
            saturation[i,1] = random.randint(0,253)
            saturation[i,2] = random.randint(0,253)

    for i in range(h):
        for j in range(w):
            a = int(objMtx[i,j])
            # imgF.itemset((i,j,0),saturation[a,0])
            # imgF.itemset((i,j,1),saturation[a,1])
            # imgF.itemset((i,j,2),saturation[a,2])
            imgF[i,j,0] = saturation[a,0]
            imgF[i,j,1] = saturation[a,1]
            imgF[i,j,2] = saturation[a,2]

    return imgF

def DrawSq(img,boxesLst):
    l = len(boxesLst)
    l = int(l)
    h,w,z = img.shape

    a = l
    color = (0,255,0)
    for i in range(a):
        if(i >= 0):
            if(boxesLst[i][2] < w and boxesLst[i][3] < w and boxesLst[i][0] < h and boxesLst[i][1] < h ):
                if(boxesLst[i][2] >0 and boxesLst[i][3] >0 and boxesLst[i][0] >0 and boxesLst[i][1] >0 ):
                    p1 = (int(boxesLst[i][2]), int(boxesLst[i][0]))#Noreste
                    p2 = (int(boxesLst[i][3]), int(boxesLst[i][0]))#Noroesste
                    p3 = (int(boxesLst[i][2]), int(boxesLst[i][1]))#Sureste
                    p4 = (int(boxesLst[i][3]), int(boxesLst[i][1]))#Suroeste
                    img = cv2.line(img, p1,p2 , color, 1)
                    img = cv2.line(img, p1,p3 , color, 1)
                    img = cv2.line(img, p2,p4 , color, 1)
                    img = cv2.line(img, p3,p4 , color, 1)

    return img

def show_image(img, name): 
    cv2.imshow(str(name), img)
    print("Click any key for next computation...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()