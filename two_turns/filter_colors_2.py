import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import chess_helper_2
import chess
import cv2
import math


BLACK = (0.0, 0.0, 0.0)
MINIMAL_PLAYER_BOARD_RATIO = 0.2
PIXELS_FOR_MAIN_COLORS = (200, 200)
PIXELS_SQUARE = (20, 20)
BLACK_NUM = 1
WHITE_NUM = 2
USER = True
RIVAL = False


class filter_colors_2:
    """
    This file is responsible for filtering and cataloguing the colors in the
    picture.
    """

    def __init__(self, im, chess_helper_2):
        self.chess_helper_2 = chess_helper_2
        self.initialize_colors(im,chess_helper_2.user_starts)
        self.initialize_board()

    def initialize_board(self):
        self.board = []
        for i in range(8):
            self.board.append([])
            for j in range(8):
                self.board[i].append(None)

    def update_board(self, last_move):
        if last_move != None:
            row1 = ord(last_move[0]) - ord('a')
            colon1 = int(last_move[1]) - 1
            row2 = ord(last_move[2]) - ord('a')
            colon2 = int(last_move[3]) - 1
            if colon1 < 6:
                self.board[row1][colon1 + 2] = None
            if colon1 < 6:
                self.board[row2][colon2 + 2] = None

    def color_dist(self, color1, color2):
        return abs(max(color1) - max(color2) - min(color2) + min(color1))

    def cmpT(self, t1, t2):
        return t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2]

    """
      gets all 4/3/2 colors in the board.
      :im initial image, that contains the relevant colors on the board.
      :return nothing
      """

    def initialize_colors(self, im):
        """
        :param im:
        :param is_me_white:
        :return black,white,user soldier color, rival soldier color:
        """
        self.set_prev_im(im)
        im_resize = cv2.resize(im, PIXELS_FOR_MAIN_COLORS)
        board_colors = self.get_board_colors(im_resize)
        user_color = self.get_player_color(im_resize, board_colors, USER)
        rival_color = self.get_player_color(im_resize, board_colors, RIVAL)
        main_colors = board_colors
        main_colors.append(user_color)
        main_colors.append(rival_color)
        self.set_colors_nums(main_colors)
        print('main colors are:')
        print(main_colors)
        self.main_colors = main_colors

    """
        gets 2 primary colors from board image.
        """

    def get_board_colors(self, im):
        # TODO fix this lines
        im_sz = len(im)
        ar = im[(im_sz // 4):(3 * im_sz // 4)]
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

    """
        gets player's colors from the board image.
        """

    def get_player_color(self, im, board_colors, player):
        is_me_white = chess_helper_2.user_starts
        # ar = np.asarray(im)
        ar = im
        # TODO fix these lines
        ar_sz = len(ar)
        if player == RIVAL:
            ar = ar[:(ar_sz // 4)]
        else:
            ar = ar[3 * (ar_sz // 4):]
        shape = ar.shape
        ar2 = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar2, 3)
        for i in range(3):
            color_dist_1 = self.color_dist(codes[i], board_colors[0])
            color_dist_2 = self.color_dist(codes[i], board_colors[1])
            if i == 0:
                max_dist = min(color_dist_1, color_dist_2)
                max_dist_index = i
            else:
                dist_i = min(color_dist_1, color_dist_2)
                if dist_i > max_dist:
                    max_dist = dist_i
                    max_dist_index = i
        player_color = codes[max_dist_index]
        num_of_player_pix = 0
        for rowidx in range(len(ar)):
            row = ar[rowidx]
            for pix in row:
                color_dist_from_player = self.color_dist(pix, player_color)
                color_dist_from_black = self.color_dist(pix, board_colors[0])
                color_dist_from_white = self.color_dist(pix, board_colors[1])
                if color_dist_from_player < color_dist_from_black and \
                        color_dist_from_player < color_dist_from_white:
                    num_of_player_pix += 1
        num_of_pix = len(ar) * len(ar[0])
        rank = num_of_player_pix / num_of_pix
        print(rank)
        if rank < MINIMAL_PLAYER_BOARD_RATIO:
            if (is_me_white and not player) or (not is_me_white and player):
                player_color = board_colors[0]
            else:
                player_color = board_colors[1]
        return player_color

    def set_colors_nums(self, main_colors):
        self.USER_NUM = 4
        self.RIVAL_NUM = 8

        # main colors order: black = 1, white = 2, user = 4, rival = 8
        if self.cmpT(main_colors[2], main_colors[0]):
            self.USER_NUM = BLACK_NUM
        elif self.cmpT(main_colors[2], main_colors[1]):
            self.USER_NUM = WHITE_NUM
        if self.cmpT(main_colors[3], main_colors[0]):
            self.RIVAL_NUM = BLACK_NUM
        elif self.cmpT(main_colors[3], main_colors[1]):
            self.RIVAL_NUM = WHITE_NUM

        self.R2W = WHITE_NUM - self.RIVAL_NUM
        self.R2B = BLACK_NUM - self.RIVAL_NUM
        self.R2U = self.USER_NUM - self.RIVAL_NUM
        self.U2R = self.RIVAL_NUM - self.USER_NUM
        self.B2R = self.RIVAL_NUM - BLACK_NUM
        self.W2R = self.RIVAL_NUM - WHITE_NUM

        return

    ###########################################################################

    """
        sets previous image - in rgb format.
        """

    def set_prev_im(self, im):
        self.prev_im = im

    """
    :im image of square after turn NOT CATALOGUED
    :square_loc location of square, in uci format
    :return binary image of relevant differences only (according to
    player/square color)
    """

    def get_square_diff(self, im, square_loc, is_source):
        is_white = self.chess_helper_2.square_color(square_loc) == chess.WHITE
        row = ord(square_loc[0]) - ord('a')
        colon = int(square_loc[1]) - 1
        after_square = cv2.resize(self.get_square_image(im, square_loc, self.chess_helper_2.user_starts), PIXELS_SQUARE)
        after_square = self.catalogue_colors(after_square, is_white)
        #maybe_before_square = self.board[row][colon]
        maybe_before_square = None
        #TODO fix it (now we are not using the saved board
        if maybe_before_square == None:
            before_square = cv2.resize(self.get_square_image(self.prev_im, square_loc, self.chess_helper_2.user_starts), PIXELS_SQUARE)
            before_square = self.catalogue_colors(before_square, is_white)
        else:
            before_square = maybe_before_square
        self.board[row][colon] = after_square
        square_diff = self.make_binary_relevant_diff_im(before_square, after_square, is_white, is_source,self.chess_helper_2.curr_player)
        square_diff_2 = np.array(square_diff)
        # square_diff_cut = cv2.cvtColor(square_diff_2, cv2.COLOR_GRAY2RGB)
        # print(square_diff_cut)
        return square_diff

    """
    :return an image with 4/3/2 colors only.
    """

    """
    returns subimage of a square in the board.
    """

    def get_square_image(self, im, loc, did_I_start):
        locidx = self.chess_helper_2.ucitoidx(loc)
        sq_sz = len(im[0]) // 8
        sq_sz_y = len(im)//8
        x = locidx[0]
        if did_I_start:
            y = 8 - locidx[1]
        else:
            y = locidx[1]
        area = (x * sq_sz, y * sq_sz_y, (x + 1) * sq_sz, (y + 1) * sq_sz_y)
        sqr_im = im[area[1]:area[3], area[0]:area[2]]
        return sqr_im

    def catalogue_colors(self, im, is_white):
        main_colors_white = self.main_colors[1:]
        main_colors_black = [self.main_colors[0]] + self.main_colors[2:]
        if is_white:
            cat_im = self.fit_colors(im, main_colors_white, is_white)
        else:
            cat_im = self.fit_colors(im, main_colors_black, is_white)
        return cat_im

    """
    :return image fit to 4 main colors.
    """

    def fit_colors(self, im, main_colors, is_white):
        new_im = []
        for rowidx in range(len(im)):
            i = 0
            row = im[rowidx]
            new_im.append([])
            for pix in row:
                min_dist = self.color_dist(pix, main_colors[0])
                if is_white:
                    new_im[rowidx].append(WHITE_NUM)
                    if not self.cmpT(main_colors[1], main_colors[0]):
                        dist = self.color_dist(pix, main_colors[1])
                        if dist < min_dist:
                            min_dist = dist
                            new_im[rowidx][i] = self.USER_NUM
                    if not self.cmpT(main_colors[2], main_colors[0]):
                        dist = self.color_dist(pix, main_colors[2])
                        if dist < min_dist:
                            new_im[rowidx][i] = self.RIVAL_NUM
                else:
                    new_im[rowidx].append(BLACK_NUM)
                    if not self.cmpT(main_colors[1], main_colors[0]):
                        dist = self.color_dist(pix, main_colors[1])
                        if dist < min_dist:
                            min_dist = dist
                            new_im[rowidx][i] = self.USER_NUM
                    if not self.cmpT(main_colors[2], main_colors[0]):
                        dist = self.color_dist(pix, main_colors[2])
                        if dist < min_dist:
                            new_im[rowidx][i] = self.RIVAL_NUM
                i += 1
        return new_im

    def make_binary_relevant_diff_im(self, im1, im2, square, is_source):
        is_white = self.chess_helper_2.square_color(square)
        RC = []
        if is_source:
            if is_white:
                RC.append(self.R2W)
            else:
                RC.append(self.R2B)
            if self.chess_helper_2.piece_color(square):
                #TODO correct chess_helper - change the colors of pieces to User and Rival
                RC.append(self.R2U)
        else:
            if is_white:
                RC.append(self.W2R)
            else:
                RC.append(self.B2R)
            if self.delayed_chess_helper.piece_color(square) or \
                    self.delayed_chess_helper.piece_color(self.chess_helper_2.get_square_bellow(square)):
                RC.append(self.U2R)
        while 0 in RC:
            RC.remove(0)
        binary_im = []
        for rowidx in range(len(im1)):
            binary_im.append([])
            for pixidx in range(len(im1[0])):
                if im2[rowidx][pixidx] - im1[rowidx][pixidx] in RC:
                    binary_im[rowidx].append(1)
                else:
                    binary_im[rowidx].append(0)
        return binary_im
