import cv2
import two_turns.filter_colors_2
import identify_board
import board_cut_fixer
import tester_helper
import numpy as np

print_and_save = True


class photos_angle_2:
    def __init__(self, hardware1, chess_helper, delay_chess_helper, self_idx):
        self.chess_helper = chess_helper
        self.delay_chess_helper = delay_chess_helper
        self.hardware = hardware1
        self.idx = self_idx
        self.boardid = identify_board.identify_board()
        self.fixer = board_cut_fixer.board_cut_fixer()

    def init_colors(self):

        cut_board_im = self.get_new_img(tester_info=(-1, self.idx))
        self.color_filter = two_turns.filter_colors_2.filter_colors_2(cut_board_im, self.chess_helper,
                                                                      self.delay_chess_helper)

    def prep_img(self):
        self.prep_im = self.hardware.get_image(self.idx)

    def get_new_img(self, tester_info=None):
        try:
            to_save = bool(tester_info)

            new_board_im = self.prep_im

            # better_cut_board_im = self.fixer.main(new_board_im)
            better_cut_board_im = new_board_im
            # TODO: switch two upper rows

            if to_save:
                move_num = tester_info[0]
                angle_idx = tester_info[1]
                tester_helper.save_bw(better_cut_board_im, 'board', move_num, angle_idx, 'second')

            return better_cut_board_im
        except:
            print("get new im failed")
            raise

    def get_square_diff(self, cut_board_im, src, is_source):
        return self.color_filter.get_square_diff(cut_board_im, src, is_source)

    def set_prev_im(self, img):
        self.fixer.set_prev_im(img)
        return self.color_filter.set_prev_im(img)

    def get_prev_im(self):
        return self.color_filter.prev_im
