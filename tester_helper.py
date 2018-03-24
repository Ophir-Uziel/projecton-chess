import cv2
import os
import errno
import numpy as np
import scipy.misc
import functools

RESULTS_DIR = 'super_tester_results'
ROWS_NUM = 9
COL_NUM = 8
PRINT = True


def line_by_line(userFileName, rivalFileName):
    moves = open("moves",'w')
    user_lines = open(userFileName).read().split("\n")
    rival_lines = open(rivalFileName).read().split("\n")
    for i in range(len(user_lines)):
        moves.write(user_lines[i]+"\n")
        moves.write(rival_lines[i]+"\n")
    moves.close()
    return moves


def save_bw(img, place, move_num, angle_idx, desc =''):
    try:
        cv2.imwrite(RESULTS_DIR + '\\' +'by_move' + '\\' + 'move_num_' + str(move_num) + '\\' + 'angle_num_' + str(angle_idx) + '\\' + place + '_' + desc + '.jpg', img)
        cv2.imwrite(RESULTS_DIR + '\\' +'by_square' + '\\' + place + '\\' + 'angle_num_' + str(angle_idx) + '\\' + str(move_num) + '_' + desc + '.jpg', img)
    except:
        pass

def save_colors(img, place, move_num, angle_idx, desc =''):
    try:
        scipy.misc.imsave(RESULTS_DIR + '\\' +'by_move' + '\\' + 'move_num_' + str(move_num) + '\\' + 'angle_num_' + str(angle_idx) + '\\' + place + '_' + desc + '.jpg', img)
        scipy.misc.imsave(RESULTS_DIR + '\\' +'by_square' + '\\' + place + '\\' + 'angle_num_' + str(angle_idx) + '\\' + str(move_num) + '_' + desc + '.jpg', img)
    except:
        pass


def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def make_board_im_helper(squares, square_ims, is_rgb = False):
    squares_im_lst = []
    for i in range(ROWS_NUM):
        for j in range(COL_NUM):
            square = chr(ord('a')+j)+str(ROWS_NUM-i)
            if square in squares:
                squares_im_lst.append(square_ims[squares.index(square)])
            else:
                squares_im_lst.append(None)
    if is_rgb:
        return make_board_im(squares_im_lst, len(square_ims[0]), len(square_ims[0][0]), 'd,d,d')
    else:
        return make_board_im(squares_im_lst, len(square_ims[0]), len(square_ims[0][0]))

def make_board_im(squares_im_lst, pic_hi, pic_wid, dtype = np.int):
    #lst of all squares. None for non relevant squares.
    if len(squares_im_lst) != ROWS_NUM*COL_NUM:
        raise Exception('make board im has failed')
    row_num = pic_hi*ROWS_NUM
    col_num = pic_wid*COL_NUM
    im = np.zeros((row_num,col_num), dtype=dtype).tolist()
    for i in range(ROWS_NUM):
        for j in range(COL_NUM):
            im_num = i*COL_NUM+j
            if squares_im_lst[im_num] is not None:
                for k in range(pic_hi):
                    im[i*pic_hi+k][j*pic_wid:(j+1)*pic_wid] = squares_im_lst[i*8+j][k]
    return im


def connect_two_ims(im,im_abv):
    im = im.tolist()
    im_abv = im_abv.tolist()
    row_num = len(im)*2
    col_num = len(im[0])
    new_im = np.zeros((row_num,col_num), np.int).tolist()
    for i in range(row_num//2):
        new_im[i] = im_abv[i]
        new_im[i+(row_num//2)] = im[i]
    return list(new_im)

def connect_two_ims_lst(im_lst, im_abv_lst):
    return list(map(connect_two_ims, im_lst, im_abv_lst))


def make_two_ims_dir(game_dir, y_or_n,counter):
    make_dir(y_or_n)
    slf_ims = os.listdir(game_dir + "\\" + "self_" + y_or_n + "_dir" )
    slf_ims = sorted(slf_ims, key=im_num)
    abv_ims = os.listdir(game_dir + "\\" + "abv_" + y_or_n + "_dir" )
    abv_ims = sorted(abv_ims, key=im_num)
    lst_idx = max(im_num(slf_ims[-1]), im_num(abv_ims[-1]))
    for j in range(lst_idx):
        im_name = "im" + str(j) + ".jpg"
        if im_name in slf_ims and im_name in abv_ims:
            im = cv2.imread(game_dir + "\\" + "self_" + y_or_n + "_dir"  + "\\"+im_name, cv2.IMREAD_GRAYSCALE).tolist()
            im_abv = cv2.imread(game_dir + "\\" + "abv_" + y_or_n + "_dir" +"\\"+ im_name, cv2.IMREAD_GRAYSCALE).tolist()
            new_im = connect_two_ims(im, im_abv)
            cv2.imwrite(y_or_n + "\\" + str(counter) + ".jpg", np.array(new_im))
            counter += 1
    return counter


def make_minimal_squares_dirs():
    make_dir(RESULTS_DIR)
    make_dir(RESULTS_DIR + '\\' + 'by_square')
    make_dir(RESULTS_DIR + '\\' + 'by_square' + '\\' + 'board')
    for k in range(2):
        make_dir(RESULTS_DIR + '\\' + 'by_square' + '\\' + 'board' + '\\' + 'angle_num_' + str(k))

def make_squares_dirs():
    make_dir(RESULTS_DIR)
    make_dir(RESULTS_DIR + '\\' + 'by_move')
    make_dir(RESULTS_DIR + '\\' + 'by_square')
    for i in range(ROWS_NUM+1):
        if i == ROWS_NUM:
            make_dir(RESULTS_DIR + '\\' + 'by_square' + '\\' + 'board')
            for k in range(2):
                make_dir(RESULTS_DIR + '\\' + 'by_square' + '\\' + 'board' + '\\' + 'angle_num_' + str(k))
        else:
            for j in range(ROWS_NUM):
                make_dir(RESULTS_DIR + '\\' + 'by_square' + '\\' + chr(ord('a')+i)+str(j+1))
                for k in range(2):
                    make_dir(RESULTS_DIR + '\\' + 'by_square' + '\\' + chr(ord('a')+i)+str(j+1) + '\\' + 'angle_num_' + str(k))

def im_num(x):
    return int(x[2:-4])

# for y_or_no in ["y", "n"]:
#     counter = 0
#     for i in range(21):
#         fold_name = "net" + str(i+1)
#         folds = os.listdir("to_train")
#         if fold_name in folds:
#             counter = make_two_ims_dir("to_train\\" + fold_name, y_or_no, counter)





