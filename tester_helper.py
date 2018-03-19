import cv2
import os
import errno

RESULTS_DIR = 'super_tester_results'




def save(img, place, move_num, angle_idx, desc = ''):
    cv2.imwrite(RESULTS_DIR + '\\' +'by_move' + '\\' + 'move_num_' + str(move_num) + '\\' + 'angle_num_' + str(angle_idx) + '\\' + place + '_' + desc + '.jpg', img)
    cv2.imwrite(RESULTS_DIR + '\\' +'by_square' + '\\' + place + '\\' + 'angle_num_' + str(angle_idx) + '\\' + str(move_num) + '_' + desc + '.jpg', img)


def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
