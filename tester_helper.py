import cv2
import os
import errno
import numpy as np
import scipy.misc

RESULTS_DIR = 'super_tester_results'
ROWS_NUM = 9
COL_NUM = 8



def save_bw(img, place, move_num, angle_idx, desc =''):
    cv2.imwrite(RESULTS_DIR + '\\' +'by_move' + '\\' + 'move_num_' + str(move_num) + '\\' + 'angle_num_' + str(angle_idx) + '\\' + place + '_' + desc + '.jpg', img)
    cv2.imwrite(RESULTS_DIR + '\\' +'by_square' + '\\' + place + '\\' + 'angle_num_' + str(angle_idx) + '\\' + str(move_num) + '_' + desc + '.jpg', img)

def save_colors(img, place, move_num, angle_idx, desc =''):
    scipy.misc.imsave(RESULTS_DIR + '\\' +'by_move' + '\\' + 'move_num_' + str(move_num) + '\\' + 'angle_num_' + str(angle_idx) + '\\' + place + '_' + desc + '.jpg', img)
    scipy.misc.imsave(RESULTS_DIR + '\\' +'by_square' + '\\' + place + '\\' + 'angle_num_' + str(angle_idx) + '\\' + str(move_num) + '_' + desc + '.jpg', img)



def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def make_board_im_helper(squares, square_ims):
    squares_im_lst = []
    for i in range(ROWS_NUM):
        for j in range(COL_NUM):
            square = chr(ord('a')+j)+str(ROWS_NUM-i)
            if square in squares:
                squares_im_lst.append(square_ims[squares.index(square)])
            else:
                squares_im_lst.append(None)
    return make_board_im(squares_im_lst, len(square_ims[0]), len(square_ims[0][0]))

def make_board_im(squares_im_lst, pic_hi, pic_wid):
    #lst of all squares. None for non relevant squares.
    if len(squares_im_lst) != ROWS_NUM*COL_NUM:
        raise Exception('make board im has failed')
    row_num = pic_hi*ROWS_NUM
    col_num = pic_wid*COL_NUM
    im = np.zeros((row_num,col_num), dtype=np.int).tolist()
    for i in range(ROWS_NUM):
        for j in range(COL_NUM):
            im_num = i*COL_NUM+j
            if squares_im_lst[im_num] is not None:
                for k in range(pic_hi):
                    im[i*pic_hi+k][j*pic_wid:(j+1)*pic_wid] = squares_im_lst[i*8+j][k]
    return im