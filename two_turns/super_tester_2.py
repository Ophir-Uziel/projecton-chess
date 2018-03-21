import game_loop_2
import os
import cv2
import errno


WITH_SAVES = True

def super_tester_2(moves_file, img_dir_lst, with_saves):

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
        if x%2 ==0:
            user_moves.append((move[0:2], move[2:4]))
        else:
            real_rival_moves.append((move[0:2], move[2:4]))
        x+=1
    moves_num = len(os.listdir(img_dir_lst[0]))-1
    angles_num = len(img_dir_lst)
    game = game_loop_2.game_loop_2(angles_num, user_moves,real_rival_moves,img_dir_lst, with_saves)
    detected_moves = []
    game.main()

def if_one_dir(dir):
    img_names = os.listdir(dir)
    sorted_img_names = sorted(img_names, key=first_2_chars)
    img_array = []
    for j in range(len(sorted_img_names)):
        if (sorted_img_names[j][-4:] == ".jpg"):
            angledir = "angle" + str(j % 2 + 1)
            cv2.imwrite(angledir + "/" + sorted_img_names[j], cv2.imread(dir + '/' +
                                                                         sorted_img_names[j], cv2.IMREAD_COLOR))


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



if_one_dir("images6")
super_tester_2("moves",["angle1","angle2"],WITH_SAVES)



