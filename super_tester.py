import game_loop
import os
import errno

WITH_SAVES = True

def super_tester(moves_file, img_dir_lst,with_saves):
    try:
        os.makedirs('super tester results')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    corrects = []
    non_corects = []
    real_moves = None
    angles_num = 2
    if moves_file is not None:
        real_moves = []
        for line in open(moves_file):
            move = line.rstrip('\n')
            real_moves.append((move[0:2], move[2:4]))
        moves_num = len(os.listdir(img_dir_lst[0]))-1
        angles_num = len(img_dir_lst)
    game = game_loop.game_loop(angles_num, real_moves,img_dir_lst, with_saves)
    detected_moves = []

    for i in range(len(real_moves)):
        detected_moves.append(game.get_new_move())
        if detected_moves[i][0] == real_moves[i][0] and detected_moves[i][1] == real_moves[i][1]:
            corrects.append(i)
        else:
            non_corects.append(i)
    print('corrects')
    print(corrects)
    print('non corrects')
    print(non_corects)

    def first_2_chars(x):
        return (x[:2])

    print('Done')

super_tester(moves_+file='moves.txt', img_dir_lst=['angle1','angle2'], with_saves=WITH_SAVES)



