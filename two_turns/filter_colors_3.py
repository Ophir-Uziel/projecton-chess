import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import chess_helper_2
import chess
import cv2
import math
import os

BLACK = (0.0, 0.0, 0.0)
MINIMAL_PLAYER_BOARD_RATIO = 0.2
MINIMAL_COLOR_DIST = 35
PIXELS_FOR_MAIN_COLORS = (400, 400)
PIXELS_SQUARE = (20, 20)
USER_NUM = 1
RIVAL_NUM = 2
USER = True
RIVAL = False
TEST = True
BLACK_TEST = (0, 0, 0)
WHITE_TEST = (255, 255, 255)


class filter_colors_3:
    """
    This file is responsible for filtering and cataloguing the colors in the
    picture.
    """

    def __init__(self, im, chess_helper_2, delay_chess_helper_2):
        self.chess_helper_2 = chess_helper_2
        self.delay_chess_helper_2 = delay_chess_helper_2
        self.initialize_colors(im)

    def color_dist(self, color1, color2):
        return abs(max(color1) - max(color2) - min(color2) + min(color1))

    def initialize_colors(self, im):
        """
        :param im:
        :return black,white,user soldier color, rival soldier color:
        """
        self.prev_im = im
        board_colors = self.get_board_colors(im)
        print('\nmain colors are:')
        print(board_colors)
        self.main_colors = board_colors
        self.rc_source = USER_NUM - RIVAL_NUM
        self.rc_target = RIVAL_NUM - USER_NUM
        if TEST:
            if self.chess_helper_2.user_starts:
                self.user_color_test = WHITE_TEST
                self.rival_color_test = BLACK_TEST
            else:
                self.user_color_test = BLACK_TEST
                self.rival_color_test = WHITE_TEST
        if self.chess_helper_2.user_starts:
            self.WHITE_NUM = USER_NUM
            self.BLACK_NUM = RIVAL_NUM
            self.user_color = board_colors[1]
            self.rival_color = board_colors[0]
        else:
            self.WHITE_NUM = RIVAL_NUM
            self.BLACK_NUM = USER_NUM
            self.user_color = board_colors[0]
            self.rival_color = board_colors[1]
        return

    def get_board_colors(self, im):
        """
        :param im:
        :return 2 primary colors from board image:
        """
        im_sz = len(im)
        ar = im[(im_sz // 3):(2 * im_sz // 3)]
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, 2)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences
        indices = [i[0] for i in
                   sorted(enumerate(-counts), key=lambda x: x[1])]
        new_indices = []
        new_codes = []
        for i in indices:
            new_codes.append(codes[i])
        if self.color_dist(new_codes[0], BLACK) < self.color_dist(new_codes[1],
                                                                  BLACK):
            new_indices.append(indices[0])
            new_indices.append(indices[1])
        else:
            new_indices.append(indices[1])
            new_indices.append(indices[0])
        return [codes[i] for i in new_indices]

    def set_prev_im(self, img):
        self.prev_im = img

    ###########################################################################

    def get_square_image(self, im, loc):
        """
        :param im:
        :param loc:
        :return subimage of a square in the board:
        """
        user_starts = self.chess_helper_2.user_starts
        locidx = self.chess_helper_2.ucitoidx(loc)
        sq_sz = len(im[0]) // 8
        sq_sz_y = len(im) // 9
        x = locidx[0]
        if user_starts:
            y = 9 - locidx[1]
        else:
            y = locidx[1]
        area = (x * sq_sz, y * sq_sz_y, (x + 1) * sq_sz, (y + 1) * sq_sz_y)
        sqr_im = im[area[1]:area[3], area[0]:area[2]]
        return sqr_im

    def fit_colors(self, im):
        """
        :param im:
        :return image fit to 4 main colors:
        """
        im_sz = len(im)
        new_im = np.ones((im_sz, im_sz), dtype=int)
        test_im = np.zeros((im_sz, im_sz), dtype='d,d,d').tolist()
        if TEST:
            for rowidx in range(im_sz):
                for pixidx in range(im_sz):
                    pix = im[rowidx][pixidx]
                    user_dist = self.color_dist(pix, self.user_color)
                    rival_dist = self.color_dist(pix, self.rival_color)
                    if user_dist < rival_dist:
                        new_im[rowidx][pixidx] = USER_NUM
                        test_im[rowidx][pixidx] = self.user_color_test
                    else:
                        new_im[rowidx][pixidx] = RIVAL_NUM
                        test_im[rowidx][pixidx] = self.rival_color_test
        else:
            for rowidx in range(im_sz):
                for pixidx in range(im_sz):
                    pix = im[rowidx][pixidx]
                    user_dist = self.color_dist(pix, self.user_color)
                    rival_dist = self.color_dist(pix, self.rival_color)
                    if user_dist < rival_dist:
                        new_im[rowidx][pixidx] = USER_NUM
                    else:
                        new_im[rowidx][pixidx] = RIVAL_NUM
        return new_im, test_im

    def make_binary_relevant_diff_im(self, im1, im2, square, is_source):
        im_sz = len(im1)
        binary_im = np.zeros((im_sz, im_sz), dtype=int)
        if is_source:
            for rowidx in range(im_sz):
                for pixidx in range(im_sz):
                    if im2[rowidx][pixidx] - im1[rowidx][pixidx] == self.rc_source:
                        binary_im[rowidx][pixidx] = 255
        else:
            for rowidx in range(im_sz):
                for pixidx in range(im_sz):
                    if im2[rowidx][pixidx] - im1[rowidx][pixidx] == self.rc_target:
                        binary_im[rowidx][pixidx] = 255
        return binary_im

    def get_square_diff(self, im, square_loc, is_source):
        """
        :param im:
        :param square_loc:
        :param is_source:
        :return binary image of relevant changes only (according alot of parameters):
        """
        after_square = cv2.resize(self.get_square_image(im, square_loc), PIXELS_SQUARE)
        after_square, after2save = self.fit_colors(after_square)
        before_square = cv2.resize(self.get_square_image(self.prev_im, square_loc), PIXELS_SQUARE)
        before_square, befor2save = self.fit_colors(before_square)
        square_diff = self.make_binary_relevant_diff_im(before_square, after_square, square_loc, is_source)
        return square_diff, befor2save, after2save


def filter_color_tester(im_bef_name, im_aft_name, loc, is_source):
    im_bef = cv2.imread(im_bef_name)
    im_aft = cv2.imread(im_aft_name)
    chess_helper = chess_helper_2.chess_helper_2(True)
    delay_chess_helper = chess_helper_2.chess_helper_2(True)
    filter = filter_colors_2(im_bef, chess_helper, delay_chess_helper)
    square_diff, befor2save, after2save = filter.get_square_diff(im_aft, loc, is_source)
    scipy.misc.imsave("test_befor.jpg", befor2save)
    scipy.misc.imsave("test_after.jpg", after2save)
    cv2.imwrite("test_diff.jpg", square_diff)
    return


def main_colors_tester(folder_name):
    chess_helper = chess_helper_2.chess_helper_2(True)
    delay_chess_helper = chess_helper_2.chess_helper_2(True)
    img_names = os.listdir(folder_name)
    img_array = []
    for j in range(len(img_names)):
        image = cv2.imread(folder_name + '/' + img_names[j], cv2.IMREAD_COLOR)
        filter = filter_colors_2(image, chess_helper, delay_chess_helper)
    return


filter_color_tester("im1.jpg","im2.jpg",'b1',False)
# main_colors_tester("images")
