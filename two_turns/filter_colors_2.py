import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import chess_helper_2
import cv2
import os
import copy

BLACK = (0, 0, 100)
MINIMAL_COLOR_DIST = 60
PIXELS_SQUARE = (20, 20)
BLACK_NUM = 1
WHITE_NUM = 2
USER = True
RIVAL = False
TEST = True
BLACK_TEST = (0, 0, 0)
WHITE_TEST = (255, 255, 255)
BLACK_LOCS = ['a5', 'b4', 'c5', 'd4', 'e5', 'f4', 'g5', 'h4']
WHITE_LOCS = ['a4', 'b5', 'c4', 'd5', 'e4', 'f5', 'g4', 'h5']
USER_LOCS = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2']
RIVAL_LOCS = ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7', 'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8']
PRINTS = True


class filter_colors_2:
    """
    This file is responsible for filtering and cataloguing the colors in the
    picture.
    """

    def __init__(self, im, chess_helper_2, delay_chess_helper_2):
        self.chess_helper_2 = chess_helper_2
        self.delay_chess_helper_2 = delay_chess_helper_2
        self.user_starts = self.chess_helper_2.user_starts
        self.bad_user = False
        self.bad_rival = False
        self.bad_board = False
        self.squares_before = {}
        self.squares_after = {}
        self.squares_before_test = {}
        self.squares_after_test = {}
        self.prev_im = im
        self.initialize_colors(im)
        self.set_colors_nums(self.main_colors)

    def color_dist(self, color1, color2):
        return abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2])

    def cmpT(self, t1, t2):
        return t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2]

    def initialize_colors(self, im):
        """
        :param im:
        :return black,white,user soldier color, rival soldier color:
        """
        main_colors = []

        black_mean_colors = []
        for loc in BLACK_LOCS:
            img = self.get_square_image(im, loc)[8:16, 8:12]
            average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
            black_mean_colors.append(average_color)
        color = tuple(map(lambda y: sum(y) / float(len(y)), zip(*black_mean_colors)))
        black_color = int(color[0]), int(color[1]), int(color[2])
        main_colors.append(black_color)

        white_mean_colors = []
        for loc in WHITE_LOCS:
            img = self.get_square_image(im, loc)[8:16, 8:12]
            average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
            white_mean_colors.append(average_color)
        color = tuple(map(lambda y: sum(y) / float(len(y)), zip(*white_mean_colors)))
        white_color = int(color[0]), int(color[1]), int(color[2])
        main_colors.append(white_color)

        user_mean_colors = []
        for loc in USER_LOCS:
            img = self.get_square_image(im, loc)[8:16, 8:12]
            average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
            user_mean_colors.append(average_color)
        color = tuple(map(lambda y: sum(y) / float(len(y)), zip(*user_mean_colors)))
        user_color = int(color[0]), int(color[1]), int(color[2])
        if PRINTS:
            print("dist( user , black ) = " + str(int(self.color_dist(black_color, user_color))))
            print("dist( user , white ) = " + str(int(self.color_dist(white_color, user_color))))
        if self.color_dist(black_color, user_color) < MINIMAL_COLOR_DIST or self.color_dist(white_color,
                                                                                            user_color) < MINIMAL_COLOR_DIST:
            if self.user_starts:
                user_color = white_color
            else:
                user_color = black_color
        main_colors.append(user_color)

        # rival_mean_colors = []
        # for loc in RIVAL_LOCS:
        #     img = self.get_square_image(im, loc)[8:16, 8:12]
        #     average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
        #     rival_mean_colors.append(average_color)
        # color = tuple(map(lambda y: sum(y) / float(len(y)), zip(*rival_mean_colors)))
        # rival_color = int(color[0]), int(color[1]), int(color[2])
        # if PRINTS:
        #     print("dist( rival , black ) = " + str(int(self.color_dist(black_color, rival_color))))
        #     print("dist( rival , white ) = " + str(int(self.color_dist(white_color, rival_color))))
        # if self.color_dist(black_color, rival_color) < MINIMAL_COLOR_DIST or self.color_dist(white_color,
        #                                                                                      rival_color) < MINIMAL_COLOR_DIST:
        #     if self.user_starts:
        #         rival_color = black_color
        #     else:
        #         rival_color = white_color
        rival_color = black_color
        # TODO change it only if we choose to play with 4 colors
        main_colors.append(rival_color)

        if (PRINTS):
            print('main colors are:')
            print(str(main_colors) + '\n')
        self.main_colors = main_colors

    def set_colors_nums(self, main_colors):
        self.USER_NUM = 4
        self.RIVAL_NUM = 8
        # main colors order: black = 1, white = 2, user = 4, rival = 8
        if self.cmpT(main_colors[2], main_colors[0]):
            self.bad_user = True
            self.USER_NUM = BLACK_NUM
            self.bad_board = True
        elif self.cmpT(main_colors[2], main_colors[1]):
            self.bad_user = True
            self.USER_NUM = WHITE_NUM
            self.bad_board = True
        if self.cmpT(main_colors[3], main_colors[0]):
            self.bad_rival = True
            self.RIVAL_NUM = BLACK_NUM
            self.bad_board = True
        elif self.cmpT(main_colors[3], main_colors[1]):
            self.bad_rival = True
            self.RIVAL_NUM = WHITE_NUM
            self.bad_board = True
        self.R2W = WHITE_NUM - self.RIVAL_NUM
        self.R2B = BLACK_NUM - self.RIVAL_NUM
        self.B2R = self.RIVAL_NUM - BLACK_NUM
        self.W2R = self.RIVAL_NUM - WHITE_NUM
        self.R2U = self.USER_NUM - self.RIVAL_NUM
        self.U2R = self.RIVAL_NUM - self.USER_NUM

        if TEST:
            self.user_color_test = (255, 100, 255)
            self.rival_color_test = (50, 100, 0)
            if self.cmpT(main_colors[2], main_colors[0]):
                self.user_color_test = BLACK_TEST
            elif self.cmpT(main_colors[2], main_colors[1]):
                self.user_color_test = WHITE_TEST
            if self.cmpT(main_colors[3], main_colors[0]):
                self.rival_color_test = BLACK_TEST
            elif self.cmpT(main_colors[3], main_colors[1]):
                self.rival_color_test = WHITE_TEST
        return

    def set_prev_im(self, img):
        self.prev_im = img

    ###########################################################################

    def get_square_image(self, im, loc):
        """
        :param im:
        :param loc:
        :return subimage of a square in the board:
        """
        locidx = self.chess_helper_2.ucitoidx(loc)
        sq_sz = len(im[0]) // 8
        sq_sz_y = len(im) // 9
        x = locidx[0]
        if self.user_starts:
            y = 9 - locidx[1]
        else:
            y = locidx[1]
        area = (x * sq_sz, y * sq_sz_y, (x + 1) * sq_sz, (y + 1) * sq_sz_y)
        sqr_im = im[area[1]:area[3], area[0]:area[2]]
        return sqr_im

    def fit_colors(self, im, loc):
        """
        :param im:
        :return image fit to 4 main colors:
        """
        im_sz = len(im)
        new_im = np.ones((im_sz, im_sz), dtype=int)
        test_im = np.zeros((im_sz, im_sz), dtype='d,d,d').tolist()
        if not TEST:
            for rowidx in range(im_sz):
                for pixidx in range(im_sz):
                    pix = im[rowidx][pixidx]
                    min_dist = self.color_dist(pix, self.main_colors[0])
                    dist = self.color_dist(pix, self.main_colors[1])
                    if dist < min_dist:
                        min_dist = dist
                        new_im[rowidx][pixidx] = WHITE_NUM
                    if self.chess_helper_2.piece_color(loc) or self.chess_helper_2.piece_color(
                            chess_helper_2.get_square_below(loc)):
                        dist = self.color_dist(pix, self.main_colors[2])
                        if dist < min_dist:
                            min_dist = dist
                            new_im[rowidx][pixidx] = self.USER_NUM
                    if not self.chess_helper_2.piece_color(loc) or not self.chess_helper_2.piece_color(
                            chess_helper_2.get_square_below(loc)):
                        dist = self.color_dist(pix, self.main_colors[3])
                        if dist < min_dist:
                            new_im[rowidx][pixidx] = self.RIVAL_NUM
        else:
            for rowidx in range(im_sz):
                for pixidx in range(im_sz):
                    pix = im[rowidx][pixidx]
                    min_dist = self.color_dist(pix, self.main_colors[0])
                    dist = self.color_dist(pix, self.main_colors[1])
                    if dist < min_dist:
                        min_dist = dist
                        new_im[rowidx][pixidx] = WHITE_NUM
                        test_im[rowidx][pixidx] = WHITE_TEST
                    if self.chess_helper_2.piece_color(loc) or self.chess_helper_2.piece_color(
                            self.chess_helper_2.get_square_below(loc)):
                        dist = self.color_dist(pix, self.main_colors[2])
                        if dist < min_dist:
                            min_dist = dist
                            new_im[rowidx][pixidx] = self.USER_NUM
                            test_im[rowidx][pixidx] = self.user_color_test
                    if not self.chess_helper_2.piece_color(loc) or not self.chess_helper_2.piece_color(
                            self.chess_helper_2.get_square_below(loc)):
                        dist = self.color_dist(pix, self.main_colors[3])
                        if dist < min_dist:
                            new_im[rowidx][pixidx] = self.RIVAL_NUM
                            test_im[rowidx][pixidx] = self.rival_color_test
        return new_im, test_im

    def make_binary_relevant_diff_im(self, im1, im2, square, is_source):
        user_is_white = self.user_starts
        is_white = self.chess_helper_2.square_color(square)
        if int(square[1]) == 9:
            above_board = True
        else:
            above_board = False
        RC = []
        if is_source:
            if above_board:
                RC.append(self.R2W)
                RC.append(self.R2B)
            elif is_white:
                RC.append(self.R2W)
            else:
                RC.append(self.R2B)
            if self.chess_helper_2.piece_color(square):
                if is_white != user_is_white:
                    if not self.bad_rival:
                        RC.append(self.R2U)
                else:
                    RC.append(self.R2U)
        else:
            if above_board:
                RC.append(self.W2R)
                RC.append(self.B2R)
            elif is_white:
                RC.append(self.W2R)
            else:
                RC.append(self.B2R)
            sq_below = self.chess_helper_2.get_square_below(square)
            if self.bad_rival and is_white != user_is_white:
                if self.delay_chess_helper_2.piece_color(square) and self.chess_helper_2.piece_color(
                        square):
                    if self.delay_chess_helper_2.piece_color(sq_below) == self.chess_helper_2.piece_color(sq_below):
                        RC.append(self.U2R)
                if self.delay_chess_helper_2.piece_color(sq_below) and self.chess_helper_2.piece_color(
                        sq_below):
                    if self.delay_chess_helper_2.piece_color(square) == self.chess_helper_2.piece_color(square):
                        RC.append(self.U2R)
            elif self.delay_chess_helper_2.piece_color(square) or self.delay_chess_helper_2.piece_color(sq_below):
                RC.append(self.U2R)

        while 0 in RC:
            RC.remove(0)
        im_sz = len(im1)
        binary_im = np.zeros((im_sz, im_sz), dtype=int)
        for rowidx in range(im_sz):
            for pixidx in range(im_sz):
                if (im2[rowidx][pixidx] - im1[rowidx][pixidx]) in RC:
                    binary_im[rowidx][pixidx] = 255
        return binary_im

    def get_square_diff(self, im, square_loc, is_source):
        """
        :param im:
        :param square_loc:
        :param is_source:
        :return binary image of relevant changes only (according alot of parameters):
        """
        if square_loc in self.squares_before.keys():
            before_square = self.squares_before[square_loc]
        else:
            before_square = cv2.resize(self.get_square_image(self.prev_im, square_loc),
                                       PIXELS_SQUARE)
            before_square, before2save = self.fit_colors(before_square, square_loc)
            self.squares_before[square_loc] = before_square
            self.squares_before_test[square_loc] = before2save
        if square_loc in self.squares_after.keys():
            after_square = self.squares_after[square_loc]
        else:
            after_square = cv2.resize(self.get_square_image(im, square_loc), PIXELS_SQUARE)
            after_square, after2save = self.fit_colors(after_square, square_loc)
            self.squares_after[square_loc] = after_square
            self.squares_after_test[square_loc] = after2save
        square_diff = self.make_binary_relevant_diff_im(before_square, after_square, square_loc, is_source)
        return square_diff

        ###########################################################################

    def update_board(self):
        self.squares_before = copy.deepcopy(self.squares_after)
        self.squares_before_test = copy.deepcopy(self.squares_after_test)
        self.squares_after = {}
        self.squares_after_test = {}

    def is_rival_equals_black(self):
        return self.bad_rival

def make_binary_relevant_diff_im_test(self, im1, im2, square, is_source):
    user_is_white = self.user_starts
    is_white = self.chess_helper_2.square_color(square)
    if int(square[1]) == 9:
        above_board = True
    else:
        above_board = False
    RC = []
    if is_source:
        if above_board:
            RC.append(self.R2W)
            RC.append(self.R2B)
        elif is_white:
            RC.append(self.R2W)
        else:
            RC.append(self.R2B)
        if self.chess_helper_2.piece_color(square):
            if is_white != user_is_white:
                if not self.bad_rival:
                    RC.append(self.R2U)
            else:
                RC.append(self.R2U)
    else:
        if above_board:
            RC.append(self.W2R)
            RC.append(self.B2R)
        elif is_white:
            RC.append(self.W2R)
        else:
            RC.append(self.B2R)
        sq_below = self.chess_helper_2.get_square_below(square)
        if self.bad_rival and is_white != user_is_white:
            if self.delay_chess_helper_2.piece_color(square) and self.chess_helper_2.piece_color(
                    square):
                if self.delay_chess_helper_2.piece_color(sq_below) == self.chess_helper_2.piece_color(sq_below):
                    RC.append(self.U2R)
            if self.delay_chess_helper_2.piece_color(sq_below) and self.chess_helper_2.piece_color(
                    sq_below):
                if self.delay_chess_helper_2.piece_color(square) == self.chess_helper_2.piece_color(square):
                    RC.append(self.U2R)
        elif self.delay_chess_helper_2.piece_color(square) or self.delay_chess_helper_2.piece_color(sq_below):
            RC.append(self.U2R)

    while 0 in RC:
        RC.remove(0)
    im_sz = len(im1)
    binary_im = np.zeros((im_sz, im_sz), dtype=int)
    for rowidx in range(im_sz):
        for pixidx in range(im_sz):
            if (im2[rowidx][pixidx] - im1[rowidx][pixidx]) in RC:
                binary_im[rowidx][pixidx] = 255
    return binary_im

def fit_colors_test(im,piece, piece_below,filter):
    """
    :param im:
    :return image fit to 4 main colors:
    """
    im_sz = len(im)
    new_im = np.ones((im_sz, im_sz), dtype=int)
    for rowidx in range(im_sz):
        for pixidx in range(im_sz):
            pix = im[rowidx][pixidx]
            min_dist = filter.color_dist(pix, filter.main_colors[0])
            dist = filter.color_dist(pix, filter.main_colors[1])
            if dist < min_dist:
                min_dist = dist
                new_im[rowidx][pixidx] = WHITE_NUM
            if piece or piece_below:
                dist = filter.color_dist(pix, filter.main_colors[2])
                if dist < min_dist:
                    min_dist = dist
                    new_im[rowidx][pixidx] = filter.USER_NUM
            if not piece or not piece_below:
                dist = filter.color_dist(pix, filter.main_colors[3])
                if dist < min_dist:
                    new_im[rowidx][pixidx] = filter.RIVAL_NUM
    return new_im

def get_square_diffs_test(im1, im2, piece_before, piece_below_before,piece_after, piece_below_after, loc, is_source):
    chesshelper = chess_helper_2.chess_helper_2()
    filter = filter_colors_2(im1,chesshelper,chesshelper)
    before_square = cv2.resize(filter.get_square_image(im1, loc), PIXELS_SQUARE)
    before_square = fit_colors_test(before_square,piece_before,piece_below_before,filter)
    after_square = cv2.resize(filter.get_square_image(im2, loc), PIXELS_SQUARE)
    after_square = fit_colors_test(after_square,piece_after,piece_below_after,filter)
    square_diff = filter.make_binary_relevant_diff_im(before_square, after_square, loc, is_source)
    return square_diff

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

    # filter_color_tester("im1.jpg","im2.jpg",'g5',False)
    # main_colors_tester("images")
