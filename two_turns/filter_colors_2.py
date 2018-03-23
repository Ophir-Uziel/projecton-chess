import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import chess_helper_2
import chess
import cv2
import math
import os
import copy

BLACK = (0, 0, 100)
MINIMAL_PLAYER_BOARD_RATIO = 0.2
MINIMAL_COLOR_DIST = 45
PIXELS_FOR_MAIN_COLORS = (400, 450)
PIXELS_SQUARE = (20, 20)
BLACK_NUM = 1
WHITE_NUM = 2
USER = True
RIVAL = False
TEST = True
BLACK_TEST = (100, 100, 100)
WHITE_TEST = (255, 255, 255)

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
        self.initialize_colors(im)

    def color_dist(self, color1, color2):
        return abs(max(color1) - max(color2) - min(color2) + min(color1))

    def cmpT(self, t1, t2):
        return t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2]

    def initialize_colors(self, im):
        """
        :param im:
        :return black,white,user soldier color, rival soldier color:
        """
        self.prev_im = im
        board_colors = self.get_board_colors(im)
        user_color = self.get_player_color(im, board_colors, USER)
        rival_color = self.get_player_color(im, board_colors, RIVAL)
        main_colors = board_colors
        main_colors.append(user_color)
        main_colors.append(rival_color)
        self.set_colors_nums(main_colors)
        print('\nmain colors are:')
        print(main_colors)
        self.main_colors = main_colors

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

    def get_player_color(self, im, board_colors, player):
        """
        :param im:
        :param board_colors:
        :param player (user or rival):
        :return player color in RGB:
        """
        black = board_colors[0]
        white = board_colors[1]
        user_starts = self.user_starts
        ar = im
        ar_sz = len(ar)
        if player == RIVAL:
            ar = ar[ar_sz // 9:(ar_sz // 3)]
            if TEST:
                cv2.imwrite("rival_board.jpg", ar)
        else:
            ar = ar[7 * (ar_sz // 9):]
            if TEST:
                cv2.imwrite("user_board.jpg", ar)
        shape = ar.shape
        ar2 = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar2, 3)
        for i in range(3):
            color_dist_1 = self.color_dist(codes[i], black)
            color_dist_2 = self.color_dist(codes[i], white)
            if i == 0:
                max_dist = min(color_dist_1, color_dist_2)
                max_dist_index = i
            else:
                dist_i = min(color_dist_1, color_dist_2)
                if dist_i > max_dist:
                    max_dist = dist_i
                    max_dist_index = i
        player_color = codes[max_dist_index]
        num_of_player_pix = 0.0
        for rowidx in range(len(ar)):
            row = ar[rowidx]
            for pix in row:
                color_dist_from_player = self.color_dist(pix, player_color)
                color_dist_from_black = self.color_dist(pix, black)
                color_dist_from_white = self.color_dist(pix, white)
                if color_dist_from_player < color_dist_from_black and \
                        color_dist_from_player < color_dist_from_white:
                    num_of_player_pix += 1
        num_of_pix = len(ar) * len(ar[0])
        rank = num_of_player_pix / num_of_pix
        if TEST:
            if player:
                print('user rank: ' + str(rank))
            else:
                print('rival rank: ' + str(rank))

        if rank < MINIMAL_PLAYER_BOARD_RATIO:
            if (user_starts and not player) or (not user_starts and player):
                player_color = black
            else:
                player_color = white
        if TEST:
            if player:
                print("dist between user to black is: " + str(self.color_dist(player_color, black)))
                print("dist between user to white is: " + str(self.color_dist(player_color, white)))
            else:
                print("dist between rival to black is: " + str(self.color_dist(player_color, black)))
                print("dist between rival to white is: " + str(self.color_dist(player_color, white)))
        if self.color_dist(player_color, black) < MINIMAL_COLOR_DIST:
            if player != self.user_starts:
                player_color = black
        elif self.color_dist(player_color, white) < MINIMAL_COLOR_DIST:
            if player == self.user_starts:
                player_color = white
        return player_color

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

    def get_square_image(self, im, loc,before):
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
        if before:
            self.squares_before[loc] = sqr_im
        else:
            self.squares_after[loc] = sqr_im
        return sqr_im

    def fit_colors(self, im):
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
                    dist = self.color_dist(pix, self.main_colors[2])
                    if dist < min_dist:
                        min_dist = dist
                        new_im[rowidx][pixidx] = self.USER_NUM
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
                    dist = self.color_dist(pix, self.main_colors[2])
                    if dist < min_dist:
                        min_dist = dist
                        new_im[rowidx][pixidx] = self.USER_NUM
                        test_im[rowidx][pixidx] = self.user_color_test
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
                if im2[rowidx][pixidx] - im1[rowidx][pixidx] in RC:
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
            before_square = cv2.resize(self.get_square_image(self.prev_im, square_loc, True),
                                       PIXELS_SQUARE)
            before_square, before2save = self.fit_colors(before_square)
            self.squares_before[square_loc] = before_square
            self.squares_before_test[square_loc] = before2save
        if square_loc in self.squares_after.keys():
            after_square = self.squares_after[square_loc]
        else:
            after_square = cv2.resize(self.get_square_image(im, square_loc, False), PIXELS_SQUARE)
            after_square, after2save = self.fit_colors(after_square)
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
