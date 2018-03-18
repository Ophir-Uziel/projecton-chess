import os
import errno
import hardware as hw
import chess_helper_2 as ch
import find_moves_rank as fm
import photos_angle_2
import chess_engine_wrapper
import gui_img_manager

"""
Main logic file.
"""
SOURCE = True
LEFT = 0
RIGHT = 1
ROWS_NUM = 8


class game_loop_2:
    def __init__(self, angles_num, user_moves_if_test=None,rival_moves_if_test=None, imgs_if_test=None, if_save_and_print=True):

        self.if_save_and_print = if_save_and_print
        self.moves_counter = -1
        self.black_im = self.create_black_im()
        if user_moves_if_test is not None:
            self.is_test = True
            self.user_moves = user_moves_if_test
            self.rival_moves = rival_moves_if_test
        else:
            self.is_test = False

        self.hardware = hw.hardware(angles_num, imgs_if_test)
        self.chesshelper = ch.chess_helper_2(ch.chess_helper_2.ME)
        self.delay_chesshelper = self.chesshelper
        self.ph_angles = []
        if not self.is_test:
            gui_img_manager.set_finished(False)

        for i in range(angles_num):
            if not self.is_test:
                gui_img_manager.set_camera(i)
                self.ph_angles.append(photos_angle_2.photos_angle_2(self.hardware, self.chesshelper,self.delay_chesshelper, i))
                self.ph_angles[i].prep_img()
            else:
                self.ph_angles.append(photos_angle_2.photos_angle_2(self.hardware, self.chesshelper, self.delay_chesshelper, i))
                self.ph_angles[i].prep_img()

        for ang in self.ph_angles:
            ang.init_colors()

        if not self.is_test:
            gui_img_manager.set_finished(True)

        self.movefinder = fm.find_moves_rank(self.chesshelper)

        self.chess_engine = chess_engine_wrapper.chess_engine_wrapper()
        self.last_move = None
        # TODO delete upper row

    def get_new_move(self):
        self.moves_counter += 1
        print("move num" + str(self.moves_counter))
        # for angle in self.ph_angles:
        #    angle.update_board(self.last_move)
        rival_move = None
        if (self.is_test):
            rival_move = self.rival_moves[self.moves_counter]
        relevant_squares = self.chesshelper.get_relevant_locations()
        sources = relevant_squares[0]
        dests = relevant_squares[1]
        pairs = []
        pairs_ranks = []
        for i in range(len(self.ph_angles)):
            gui_img_manager.set_camera(i)
            self.ph_angles[i].prep_img()
        for i in range(len(self.ph_angles)):
            while True:
                try:
                    gui_img_manager.set_camera(i)
                    pairs_and_ranks = self.check_one_direction(sources, dests, angle_idx=i)
                    break
                except:
                    print("id error plz take another photo k thnx")
                    gui_img_manager.reset_images(i)
                    self.ph_angles[i].prep_img()
            pairs = pairs + pairs_and_ranks[0]
            pairs_ranks = pairs_ranks + pairs_and_ranks[1]
        best_pair_idx = [i for i in range(len(pairs_ranks)) if pairs_ranks[i] == max(pairs_ranks)][0]
        move = pairs[best_pair_idx]

        # if self.if_save_and_print:
        if True:
            # TODO change the fucking if
            print("detected_move")
            print(move)
            print('rival_move')
            print(rival_move)
        if self.is_test:
            move = rival_move
        self.last_move = move
        return move

    def check_one_direction(self, sources, dests, angle_idx):
        make_dir('super tester results/move_num_' + str(self.moves_counter) + '/angle_num_' + str(angle_idx))
        angle_dir = 'super tester results/move_num_' + str(self.moves_counter) + '/angle_num_' + str(angle_idx) + '/'
        rival_move = None
        angle = self.ph_angles[angle_idx]
        cut_board_im = angle.get_new_img(angle_dir)
        if self.if_save_and_print:
            print("angle_num_" + str(angle_idx))

            print("sources are:")
            print(sources)

            print("destinations are:")
            print(dests)

        else:
            rival_move = None
            angle_dir = None

        if (self.is_test):
            rival_move = self.rival_moves[self.moves_counter]

        sourcesims, sourcesabvims = self.get_diff_im_and_dif_abv_im_list(sources, cut_board_im, angle,
                                                                         SOURCE)
        destsims, destsabvims = self.get_diff_im_and_dif_abv_im_list(dests, cut_board_im, angle,
                                                                     not SOURCE)

        pairs, pairs_rank = self.movefinder.get_move(sources, sourcesims, sourcesabvims,
                                                     dests, destsims, destsabvims, rival_move, angle_dir)

        ### save prev picture ###
        angle.set_prev_im(cut_board_im)

        return pairs, pairs_rank

    def get_diff_im_and_dif_abv_im_list(self, locs, cut_board_im, angle, is_source):
        angle_dir = 'super tester results/move_num_' + str(self.moves_counter) + '/angle_num_' + str(angle.idx) + '/'
        locssims = []
        locsabvims = []
        for loc in locs:
            abv_loc = self.get_abv_loc(loc)
            bel_loc = self.get_bel_loc(loc)
            diff_im = angle.get_square_diff(cut_board_im, loc, is_source)
            if abv_loc:
                diff_abv_im = angle.get_square_diff(cut_board_im, abv_loc, is_source)
            else:
                diff_abv_im = self.black_im
                # if self.if_save_and_print:
                #   if loc == rival_move[0] or loc == rival_move[1] or bel_loc == rival_move[0] or bel_loc == rival_move[1]:
                #      cv2.imwrite(angle_dir + loc + '.jpg', diff_im)
            locssims.append(diff_im)
            locsabvims.append(diff_abv_im)
        return locssims, locsabvims

    def create_black_im(self):
        black_im = []
        for i in range(20):
            black_im.append([])
            for j in range(20):
                black_im[i].append(0)
        return black_im

    def main(self):
        last_move = None
        while True:
            gui_img_manager.set_finished(False)
            self.best_move = self.chess_engine.get_best_move(last_move)
            print("I recommend: " + self.best_move)
            if not self.is_test:
                self.hardware.player_indication(self.best_move)
            self.delay_chesshelper = self.chesshelper
            if self.is_test:
                self.best_move = self.user_moves[self.moves_counter]
                print("sorry, I changed my mind. play" + self.best_move)
            self.chesshelper.do_turn(self.best_move[0], self.best_move[1])
            last_move = self.get_new_move()
            self.chesshelper.do_turn(last_move[0], last_move[1])
            gui_img_manager.set_finished(True)

def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise