import copy
import cv2
import math
import identify_board
#import board_cut_fixer

DEBUG = False

#####number of changes to be an legal image
CHANGE_COLOR_SAF = 3
CHANGE_DIFF = 6
BOTTOM_MAX_CHANGE_BEL = 3
TOP_MAX_CHANGE_ABV = 3


#####sizes of the window in the convolution Hor
PartOfWindowHeightHor = 24
PartOfWindowLengthUp = 12
PartOfWindowLengthDown = 12
ConvSkipX = 4
ConvSkipY = 4

#####sizes of the window in the convolution Ver
PartOfWindowHeightVer = 12
PartOfWindowLengthVer = 24
ConvSkipX = 4
ConvSkipY = 4


def doConvUp(img):
    window_length = len(img) // PartOfWindowLengthUp
    window_height = len(img[0]) // PartOfWindowHeightHor

    changeCounterAbv = 0

    lastcolorFlag = -1  ##zero if last is white, 1 if black

    for i in range(0, len(img) - window_length - 1, window_length//2):
        whitepix = 0
        blackpix = 0
        for j in range(i, i + window_length):
            for k in range(window_height):
                if k >= len(img) or j >= len(img[0]):
                    continue
                if (img[k][j] > 0):
                    whitepix = whitepix + 1
                else:
                    blackpix = blackpix + 1
                if DEBUG:
                    img[window_height][j] = 255-img[window_height][j]
        if whitepix > blackpix:
            specificcolorFlag = 1  ##zero if last is white, 1 if black
        else:
            specificcolorFlag = 0
        if lastcolorFlag == -1:
            lastcolorFlag = specificcolorFlag
        if specificcolorFlag != lastcolorFlag:
            if DEBUG:
                cv2.imshow('hi', img)
                k = cv2.waitKey(0)
                print(i)
                print ("color changed")
            changeCounterAbv = changeCounterAbv + 1
            lastcolorFlag = specificcolorFlag

    return changeCounterAbv

def doConvDown(img):
    window_length = len(img) // PartOfWindowLengthDown
    window_height = len(img[0]) // PartOfWindowHeightHor

    changeCounterAbv = 0

    lastcolorFlag = -1  ##zero if last is white, 1 if black

    for i in range(0, len(img) - window_length - 1, window_length//2):
        whitepix = 0
        blackpix = 0
        for j in range(i, i + window_length):
            for k in range(len(img[0])-window_height,len(img[0])):
                if k >= len(img) or j >= len(img[0]):
                    break
                if (img[k][j] > 0):
                    whitepix = whitepix + 1
                else:
                    blackpix = blackpix + 1
                if DEBUG:
                    img[len(img)-window_height][j] = 255-img[len(img)-window_height][j]
        if whitepix > blackpix:
            specificcolorFlag = 1  ##zero if last is white, 1 if black
        else:
            specificcolorFlag = 0
        if lastcolorFlag == -1:
            lastcolorFlag = specificcolorFlag
        if specificcolorFlag != lastcolorFlag:
            if DEBUG:
                cv2.imshow('hi', img)
                k = cv2.waitKey(0)
                print ("color changed")
            changeCounterAbv = changeCounterAbv + 1
            lastcolorFlag = specificcolorFlag

    return changeCounterAbv

def doConvLeft(img):
    window_length = len(img) // PartOfWindowLengthVer
    window_height = len(img[0]) // PartOfWindowHeightVer

    changeCounterAbv = 0

    lastcolorFlag = -1  ##zero if last is white, 1 if black

    for i in range(0, len(img[0]) - window_height - 1, window_height//2):
        whitepix = 0
        blackpix = 0
        for j in range(i, i + window_height):
            for k in range(window_length):
                if k >= len(img) or j >= len(img[0]):
                    break
                if (img[j][k] > 0):
                    whitepix = whitepix + 1
                else:
                    blackpix = blackpix + 1
                if DEBUG:
                    img[j][window_length] = 255-img[j][window_length]
        if whitepix > blackpix:
            specificcolorFlag = 1  ##zero if last is white, 1 if black
        else:
            specificcolorFlag = 0
        if lastcolorFlag == -1:
            lastcolorFlag = specificcolorFlag
        if specificcolorFlag != lastcolorFlag:
            if DEBUG:
                cv2.imshow('hi', img)
                k = cv2.waitKey(0)
                print ("color changed")
            changeCounterAbv = changeCounterAbv + 1
            lastcolorFlag = specificcolorFlag

    return changeCounterAbv

def doConvRight(img):
    window_length = len(img) // PartOfWindowLengthVer
    window_height = len(img[0]) // PartOfWindowHeightVer

    changeCounterAbv = 0

    lastcolorFlag = -1  ##zero if last is white, 1 if black

    for i in range(0, len(img[0]) - window_height - 1, window_height//2):
        whitepix = 0
        blackpix = 0
        for j in range(i, i + window_height):
            for k in range(len(img[0])-window_length,len(img[0])):
                if k >= len(img) or j >= len(img[0]):
                    continue
                if (img[j][k] > 0):
                    whitepix = whitepix + 1
                else:
                    blackpix = blackpix + 1
                if DEBUG:
                    img[j][len(img[0])-window_length] = 255-img[j][len(img[0])-window_length]
        if whitepix > blackpix:
            specificcolorFlag = 1  ##zero if last is white, 1 if black
        else:
            specificcolorFlag = 0
        if lastcolorFlag == -1:
            lastcolorFlag = specificcolorFlag
        if specificcolorFlag != lastcolorFlag:
            if DEBUG:
                cv2.imshow('hi', img)
                k = cv2.waitKey(0)
                print ("color changed")
            changeCounterAbv = changeCounterAbv + 1
            lastcolorFlag = specificcolorFlag

    return changeCounterAbv

def board_cut_chacker(threshimg):
    IsCutExect= doConvUp(threshimg) > CHANGE_COLOR_SAF and doConvDown(
        threshimg) > CHANGE_COLOR_SAF and doConvLeft(threshimg) > CHANGE_COLOR_SAF and \
                doConvRight(threshimg) > CHANGE_COLOR_SAF
    return IsCutExect

def test(foldername):
    # get lines from image, and edge-image
    id = identify_board.identify_board()
    for j in range(1):
            edgeim , realimg = id.get_image_from_filename(
                foldername+"\\fixed\\"+str(
                j)+".jpg",False)
            gaus = id.get_image_from_img(realimg,False,True)
            cv2.imwrite(foldername+"\\fixed\\"+str(300)+".jpg",gaus)

            if DEBUG:
                cv2.imshow("",realimg)
                k=cv2.waitKey(0)
                cv2.imshow("",gaus)
                k=cv2.waitKey(0)
            IsCutExect = board_cut_chacker(gaus)
            print((str(IsCutExect)) + " " + str(j))
    #    except:
     #       print(str(j)+" failed")

test('fixed2')