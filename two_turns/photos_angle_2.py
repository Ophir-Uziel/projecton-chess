import cv2
import two_turns.filter_colors_2
import identify_board
import board_cut_fixer
import tester_helper
import numpy as np
print_and_save = True


class photos_angle_2:
    def __init__(self, hardware1, chess_helper,delay_chess_helper, self_idx):
        self.chess_helper = chess_helper
        self.delay_chess_helper = delay_chess_helper
        self.hardware = hardware1
        self.idx = self_idx
        self.boardid = identify_board.identify_board()
        self.fixer = board_cut_fixer.board_cut_fixer()

    def init_colors(self):

        cut_board_im = self.get_new_img()
        self.color_filter = two_turns.filter_colors_2.filter_colors_2(cut_board_im, self.chess_helper, self.delay_chess_helper)

    def prep_img(self):
        self.prep_im = self.hardware.get_image(self.idx)

    def get_new_img(self, tester_info=None):
        to_save = bool(tester_info)

        new_board_im = self.prep_im

        cut_board_im, edges = self.boardid.main(new_board_im)
        #edges, cut_board_im = self.boardid.process_im(cut_board_im, should_cut=True)
        better_cut_board_im = self.fixer.main(cut_board_im, edges)

        if to_save:
            move_num = tester_info[0]
            angle_idx = tester_info[1]
            tester_helper.save(cut_board_im, 'board', move_num, angle_idx, 'first')
            tester_helper.save(better_cut_board_im, 'board', move_num, angle_idx, 'second')

        return better_cut_board_im

    def get_square_diff(self, cut_board_im, src, is_source):
        return self.color_filter.get_square_diff(cut_board_im, src, is_source)



    def set_prev_im(self, img):
        return self.color_filter.set_prev_im(img)

    def get_prev_im(self):
        return self.color_filter.prev_im
