import game_loop_2
import os
import cv2
import errno


WITH_SAVES = True

def super_tester_2(user_moves_file, rival_moves_file, img_dir_lst, with_saves):

    corrects = []
    non_corects = []
    user_moves = []
    real_rival_moves = []
    for line in open(user_moves_file):
        move = line.rstrip('\n')
        user_moves.append((move[0:2], move[2:4]))
    for line in open(rival_moves_file):
        move = line.rstrip('\n')
        real_rival_moves.append((move[0:2], move[2:4]))
    moves_num = len(os.listdir(img_dir_lst[0]))-1
    angles_num = len(img_dir_lst)
    game = game_loop_2.game_loop_2(angles_num, user_moves,real_rival_moves,img_dir_lst, with_saves)
    detected_moves = []
    game.main()

def helper(dir):
    img_lst = os.listdir(dir)
    os.mkdir("angle1")
    os.mkdir("angle2")
    sorted_img_names = sorted(img_lst, key=first_2_chars)
    for i in range(len(sorted_img_names)//2):
        cv2.imwrite("angle1\\" +sorted_img_names[2*i], cv2.imread(dir +"\\" +sorted_img_names[2*i]))
        cv2.imwrite("angle2\\" +sorted_img_names[2*i+1], cv2.imread(dir +"\\" +sorted_img_names[2*i+1]))


def first_2_chars(x):
    return int(x[0:-4])
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

super_tester_2( "user_moves.txt","rival_moves.txt",["angle1","angle2"],WITH_SAVES)



