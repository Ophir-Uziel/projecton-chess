import os
import errno
import hardware as hw
import chess_helper_2 as ch
import find_move_rank2 as fm
import photos_angle_2
import chess_engine_wrapper
import gui_img_manager
import cv2
import numpy as np
import tester_helper

"""
Main logic file.
"""
SOURCE = True
LEFT = 0
RIGHT = 1
ROWS_NUM = 8
RESULTS_DIR = 'professional games 3\\super_tester_results'

PRINTS = True

MAX_DIFF_RATIO = 0.15

class game_loop_2:
    def __init__(self, angles_num, user_moves_if_test=None,rival_moves_if_test=None, imgs_if_test=None, if_save_and_print=True, net_dir_name = None):
        global RESULTS_DIR
        RESULTS_DIR += str(net_dir_name)
        tester_helper.RESULTS_DIR += str(net_dir_name)
        if net_dir_name:
            net_dir_name = os.path.join(RESULTS_DIR,net_dir_name)
        else:
            net_dir_name = None
        self.if_save_and_print = if_save_and_print
        if self.if_save_and_print:
            tester_helper.make_minimal_squares_dirs()
        self.moves_counter = 0
        self.last_move = None

        if user_moves_if_test is not None:
            self.is_test = True
            self.net_dir_name = net_dir_name
            self.user_moves = user_moves_if_test
            self.rival_moves = rival_moves_if_test
            if imgs_if_test:
                self.is_live_test = False
            else:
                self.is_live_test = True
        else:
            self.is_test = False
            self.is_live_test = False
            self.net_dir_name = None
            self.rival_moves = None

        self.hardware = hw.hardware(angles_num, imgs_if_test)
        self.chesshelper = ch.chess_helper_2(ch.chess_helper_2.USER)
        self.delay_chesshelper = ch.chess_helper_2(ch.chess_helper_2.USER)
        self.ph_angles = []
        if not self.is_test:
            gui_img_manager.set_finished(False)

        for i in range(angles_num):
            if not self.is_test:
                gui_img_manager.set_camera(i)
            self.ph_angles.append(photos_angle_2.photos_angle_2(self.hardware, self.chesshelper, self.delay_chesshelper, i))

        for ang in self.ph_angles:
            ang.init_colors()

        if not self.is_test:
            gui_img_manager.set_finished(True)

        self.movefinder = fm.find_moves_rank(self.chesshelper, self.net_dir_name)
        self.chess_engine = chess_engine_wrapper.chess_engine_wrapper()


    def main(self):
        while True:
            if self.is_test and (self.moves_counter >= len(self.user_moves) \
                    or self.moves_counter >= len(self.rival_moves)):
                if (PRINTS):
                    print('Done')
                if self.is_live_test or not self.is_test:
                    self.hardware.close()
                break
            gui_img_manager.set_finished(False)
            if self.is_test:
                print(self.user_moves[self.moves_counter])
                print(self.rival_moves[self.moves_counter])
            self.play_user_turn()
            if self.if_save_and_print:
                if (PRINTS):
                    print(self.chesshelper.board)
            self.play_rival_move()
            if self.if_save_and_print:
                if (PRINTS):
                    print(self.chesshelper.board)
            gui_img_manager.set_finished(True)

    def play_user_turn(self):
        self.best_move = self.chess_engine.get_best_move(self.last_move)
        if self.if_save_and_print:
            print("I recommend: " + self.best_move)
        self.hardware.player_indication(self.best_move)
        if self.is_test:
            self.best_move = self.user_moves[self.moves_counter]
            print("sorry, I changed my mind. play" + str(self.best_move))

        self.chesshelper.do_turn(self.best_move[0:2], self.best_move[2:4])

    def play_rival_move(self):
        if self.if_save_and_print:
            print("\n" + "move num" + str(self.moves_counter))

        # real move: only if test
        rival_move = None
        if (self.is_test):
            rival_move = self.rival_moves[self.moves_counter]

        # get relevant squares
        relevant_squares = self.chesshelper.get_relevant_locations()
        sources = relevant_squares[0]
        dests = relevant_squares[1]
        pairs = []
        pairs_ranks = []

        #get two images per iteration, untill has one abs good move
        cnt = 0
        to_continue = True
        while to_continue and (len(pairs) == 0 or len(pairs_ranks) == 0):
            try:
                if self.is_test and not self.is_live_test:
                    to_continue = False
                for i in range(len(self.ph_angles)):
                    if cnt > 0:
                        print("id error plz take another photo k thnx")
                    gui_img_manager.set_camera(i)

                    if self.is_live_test:
                        while True:
                            pairs_and_ranks = self.check_one_direction(sources, dests, angle_idx=i)
                            if not (len(pairs_and_ranks[0]) == 0 and len(
                                            pairs_and_ranks[1]) == 0):
                                break
                    else:
                        pairs_and_ranks = self.check_one_direction(sources, dests, angle_idx=i)
                    gui_img_manager.reset_images(i)
                    pairs = pairs + pairs_and_ranks[0]
                    pairs_ranks = pairs_ranks + pairs_and_ranks[1]
                cnt += 1
                best_pair_idx = [i for i in range(len(pairs_ranks)) if pairs_ranks[i] == max(pairs_ranks)][0]
                move = pairs[best_pair_idx]
            except:
                move = ' both direction failed'
                print(move)

        if self.if_save_and_print:
            if(PRINTS):
                print("detected_move")
                print(move)
                print('rival_move')
                print(rival_move)

        if self.is_test:
            move = rival_move
        self.last_move = move[0] + move[1]
        self.chesshelper.do_turn(move[0], move[1])

        # delayed helper do his turn now for filter_colors needs
        self.delay_chesshelper.do_turn(self.best_move[0:2],self.best_move[2:4])
        self.delay_chesshelper.do_turn(move[0], move[1])
        self.moves_counter += 1
        return move

    def check_one_direction(self, sources, dests, angle_idx):
        try:
            self.ph_angles[angle_idx].prep_img()
            if self.if_save_and_print:
                if (PRINTS):
                    print("angle_num_" + str(angle_idx))
                tester_helper.make_dir(RESULTS_DIR + '\\' + 'by_move/move_num_' + str(self.moves_counter) + '/angle_num_' + str(angle_idx))

            rival_move = None
            if (self.is_test):
                rival_move = self.rival_moves[self.moves_counter]

            angle = self.ph_angles[angle_idx]
            cut_board_im = angle.get_new_img(tester_info=(self.moves_counter, angle_idx))
            sourcesims, sourcesabvims, srcdiff = self.get_diff_im_and_dif_abv_im_list(sources, cut_board_im, angle,
                                                                             SOURCE)
            destsims, destsabvims, dstdiff = self.get_diff_im_and_dif_abv_im_list(dests, cut_board_im, angle,
                                                                         not SOURCE)
            difftot = (srcdiff + dstdiff)/(160*180)

            if difftot>MAX_DIFF_RATIO: ## too much white in img
                print("img diff was too big!")
                raise Exception()
            if self.is_test:
                tester_info = rival_move, self.moves_counter,angle_idx
            else:
                tester_info = None
            pairs, pairs_rank = self.movefinder.get_move(sources, sourcesims, sourcesabvims, dests, destsims, destsabvims,
                                                         tester_info = tester_info)
            board_before = angle.get_board_test(True)
            angle.update_board()
            if self.if_save_and_print:
                above_src = [self.chesshelper.get_square_above(src) for src in sources]
                above_trgt = [self.chesshelper.get_square_above(trgt) for trgt in dests]
                source_big_im = tester_helper.make_board_im_helper(sources+above_src,sourcesims+sourcesabvims)
                tester_helper.save_bw(np.array(source_big_im), "board", self.moves_counter, angle_idx, "src_big_im")
                target_big_im = tester_helper.make_board_im_helper(dests+above_trgt,destsims+destsabvims)
                tester_helper.save_bw(np.array(target_big_im), "board", self.moves_counter, angle_idx, "trgt_big_im")
                before_big_im = tester_helper.make_board_im_helper(list(board_before.keys()),list(board_before.values()),True)
                tester_helper.save_colors(before_big_im,"board",self.moves_counter,angle_idx,"berko")

            ### save prev picture ###
            angle.set_prev_im(cut_board_im)

            return (pairs, pairs_rank)
        except:
            print("angle " + str(angle_idx) + " failed")
            return ([], [])

    def get_diff_im_and_dif_abv_im_list(self, locs, cut_board_im, angle, is_source):
        try:
            diffarea = 0
            diffcount = 0
            locsims = []
            locsabvims = []
            for loc in locs:
                abv_loc = self.chesshelper.get_square_above(loc)
                diff_im = angle.get_square_diff(cut_board_im, loc, is_source)
                diff_abv_im = angle.get_square_diff(cut_board_im, abv_loc, is_source)
                locsims.append(diff_im)
                locsabvims.append(diff_abv_im)
                diffarea+=cv2.countNonZero(diff_im)
                diffcount+=1
                if not abv_loc in locs:
                    diffarea += cv2.countNonZero(diff_abv_im)
                    diffcount += 1
            diffnormalised = 0
            if diffcount>0:
                diffnormalised = diffarea*64/diffcount
            return locsims, locsabvims, diffnormalised

        except Exception as e:
            if is_source:
                b_val ="source"
            else:
                b_val="target"
            print("get_dif_im ("+b_val+")_failed" )
            print(str(e))
            raise


game = game_loop_2(2)
game.main()




