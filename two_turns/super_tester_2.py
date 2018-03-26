import game_loop_2
import os
import cv2
import errno
import tester_helper
import shutil
import hardware
import board_cut_fixer
WITH_SAVES = True
from two_turns import photos_angle_2
from  two_turns import chess_helper_2

def super_tester_2(moves_file, img_dir_lst, with_saves, net_idx = None):
    if net_idx:
        net_dir_name = "net" +str(net_idx)
        make_dir(game_loop_2.RESULTS_DIR+ net_dir_name+"\\"+net_dir_name)
    else:
        net_dir_name = None
    shutil.copyfile(moves_file,game_loop_2.RESULTS_DIR+
                    net_dir_name+"\\moves.txt")
    corrects = []
    non_corects = []
    user_moves = []
    real_rival_moves = []
    # for line in open(user_moves_file):
    #     move = line.rstrip('\n')
    #     user_moves.append((move[0:2], move[2:4]))
    x = 0
    for line in open(moves_file):
        move = line.rstrip('\n')
        if len(move) == 4:
            if x%2 ==0:
                user_moves.append((move[0:2], move[2:4]))
            else:
                real_rival_moves.append((move[0:2], move[2:4]))
        elif len(move) != 0:
            raise Exception("illegal move:" + move)
        x+=1
    # angles_num = len(img_dir_lst)
    angles_num = 2
    game = game_loop_2.game_loop_2(angles_num, user_moves,
                                            real_rival_moves,
                                    img_dir_lst, with_saves, net_dir_name)
    detected_moves = []
    game.main()

def berkos_tester(fold_name):
    make_dir("berkos")
    if fold_name in os.listdir("berkos"):
        raise Exception("change the name of tje folder!")
    else:
        make_dir("berkos\\" + fold_name)
    ch = chess_helper_2.chess_helper_2(True)
    hw = hardware.hardware(2)
    cnt = 0
    while True:
        cnt += 1
        for i in range(2):
            while True:
                try:
                    ph = photos_angle_2.photos_angle_2(hw,ch,ch,i)
                    ph.prep_img()
                    img = ph.get_new_img()
                    cv2.imwrite("berkos\\" + fold_name +"\\" + str(cnt) +
                                "_" +
                                str(i) +".jpg", img)
                    break
                except:
                    print("pls take_another_img")

def berkos_tester_2(fold_name):
    make_dir("berkos")
    if fold_name in os.listdir("berkos"):
        raise Exception("change the name of tje folder!")
    else:
        make_dir("berkos\\" + fold_name)
    ch = chess_helper_2.chess_helper_2(True)
    hw = hardware.hardware(1)
    cnt = 0
    while True:
        cnt += 1
        for i in range(1):
            while True:
                try:
                    ph = photos_angle_2.photos_angle_2(hw, ch, ch, i)
                    ph.prep_img()
                    img = ph.get_new_img()
                    cv2.imwrite("berkos\\" + fold_name + "\\" + str(cnt) +
                                "_" +
                                str(i) + ".jpg", img)
                    break
                except:
                    print("pls take_another_img")


def if_one_dir(dir):
    img_names = os.listdir(dir)
    sorted_img_names = sorted(img_names, key=first_2_chars)
    img_array = []
    for j in range(len(sorted_img_names)):
        if (sorted_img_names[j][-4:] == ".jpg"):
            angledir = "angle" + str(j % 2 + 1)
            cv2.imwrite(angledir + "/" + sorted_img_names[j], cv2.imread(dir + '/' +
                                                                         sorted_img_names[j], cv2.IMREAD_COLOR))
def if_one_dir_new(dir):
    img_names = os.listdir(dir)
    make_dir("angle0")
    make_dir("angle1")
    for img_name in img_names:
        if img_name[-4:] == ".jpg":
            dir_name = "angle" + img_name[0]
            im = cv2.imread(dir + "/" + img_name)
            cv2.imwrite(dir_name + "/" + img_name, im)

def first_2_chars(x):
    return int(x[3:-4])
    # for i in range(moves_num):
    #     detected_moves.append(game.get_new_move())
    #     if detected_moves[i][0] == real_rival_moves[i][0] and detected_moves[i][1] == real_rival_moves[i][1]:
    #         corrects.append(i)
    #     else:
    #         non_corects.append(i)
    # print('corrects')
    # print(corrects)
    # print('non corrects')
    # print(non_corects)
    # print('Done')

# gameloop = game_loop_2.game_loop_2(angles_num = 2)
# gameloop.main()

def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise





# if_one_dir_new("game")
IDX = 33
# super_tester_2("move_files\\moves"+str(IDX), None, WITH_SAVES,IDX)





berkos_tester("1_4_3")