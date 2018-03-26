import os
import cv2
from random import *
dirname = "data"
destname = os.path.join(dirname, 'to_classify')
imidxs = []
if not os.path.exists(destname):
    os.makedirs(destname)
for i in range(3):
    for j in range(3):
        for k in range(3):
            for i2 in range(3):
                for j2 in range(3):
                    for k2 in range(3):
                        for three_seven in range(2):
                            for r_l in range(2):
                                if i == 0:
                                    middle_piece_before = False
                                elif i==1:
                                    middle_piece_before = True
                                else:
                                    middle_piece_before = None

                                if i2 == 0:
                                    middle_piece_after= False
                                elif i2 == 1:
                                    middle_piece_after = True
                                else:
                                    middle_piece_after = None

                                if j == 0:
                                    below_piece_before = False
                                elif j==1:
                                    below_piece_before = True
                                else:
                                    below_piece_before = None

                                if j2 == 0:
                                    below_piece_after= False
                                elif j2 == 1:
                                    below_piece_after = True
                                else:
                                    below_piece_after = None

                                if k == 0:
                                    above_piece_before = False
                                elif k == 1:
                                    above_piece_before = True
                                else:
                                    above_piece_before = None

                                if k2 == 0:
                                    above_piece_after = False
                                elif k2 == 1:
                                    above_piece_after = True
                                else:
                                    above_piece_after = None

                                if three_seven == 0:
                                    three_seven_marker = 3
                                else:
                                    three_seven_marker = 7

                                if r_l == 0:
                                    right_left = 'r'
                                else:
                                    right_left = 'l'

                                foldername1 = str(i)+str(j)+str(k)+str(
                                    three_seven_marker) + right_left
                                folderdir1 = os.path.join(dirname, foldername1)

                                foldername2 = str(i) + str(j) + str(k) + str(
                                    three_seven_marker) + right_left
                                folderdir2 = os.path.join(dirname, foldername2)

                                ims1 = []
                                ims2 = []
                                for filename1 in os.listdir(folderdir1):
                                    if filename1.endswith(
                                            ".jpg"):
                                        ims1.append(cv2.imread(filename1,
                                                               cv2.IMREAD_COLOR))

                                for filename2 in os.listdir(
                                        folderdir2):
                                    if filename2.endswith(
                                            ".jpg"):
                                        ims2.append(cv2.imread(filename2,
                                                               cv2.IMREAD_COLOR))
                                for im1 in ims1:
                                    for im2 in ims2:
                                        diff1 = get_square_diffs_test(im1, im2,
                                                                   middle_piece_before,
                                                                  below_piece_before,
                                                                      middle_piece_after,
                                                                  below_piece_after, loc,
                                                                  is_source)

                                        diff2 = get_square_diffs_test(im1, im2,
                                                                      above_piece_before,
                                                                      middle_piece_before,
                                                                      above_piece_after,
                                                                      middle_piece_after,
                                                                      loc,
                                                                      is_source)
                                        while True:
                                            imgidx = randint(0,999999999)
                                            if imgidx not in imidxs:
                                                cv2.imwrite(os.path.join(destname,
                                                                         str(
                                                                             imgidx)+".jpg" ),diff1)
                                                imidxs.append(imgidx)
                                                break

                                        while True:
                                            imgidx = randint(0, 999999999)
                                            if imgidx not in imidxs:
                                                cv2.imwrite(
                                                    os.path.join(destname,
                                                                 str(
                                                                     imgidx) + ".jpg"),
                                                    diff2)
                                                imidxs.append(imgidx)
                                                break
