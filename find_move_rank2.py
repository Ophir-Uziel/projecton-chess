import numpy
import chess_helper
import cv2
import numpy as np
import tester_helper
import scikit_learn


"""
This file is responsible for identifying if a move has been made in a square.
"""


mindensity=0.1
maxdensity=3.5
minyahas=0.5
maxyahas=1
normalizing_factor_density = (((maxdensity+mindensity)*0.5)**2/2)+(
    maxdensity*mindensity*(-0.5)+0.2)
normalizing_factor_yahas = (((maxyahas+minyahas)**2)/4)+(0.9-maxyahas*minyahas)
parameterOfSize=0.45 ##the minimal size of white that should be in an image
#if its lower than that, we cancel the image

WHITE = 1
BLACK = 0

CROP_RT = 6
SIDE_CROP_RT = 10
INIT_HI_RT = 20

MAX_RT = 1.5
WINDOW_RT = 10
WHITE_DEF_RT = 2
MIN_WHITE_RT = 5

PRINTS = False
SAVE_ALLOT = False

class find_moves_rank:


    def __init__(self, chess_helper, net_dir_name = None):
        self.net = scikit_learn.read_net(scikit_learn.NET_FILE)
        self.neuron_counter = 0
        if SAVE_ALLOT:
            tester_helper.make_squares_dirs()
        self.chess_helper = chess_helper
        self.mistake_idxes = []
        if net_dir_name:
            self.net_dir_name = net_dir_name
            tester_helper.make_dir(net_dir_name + "\\" + 'self_y_dir')
            tester_helper.make_dir(net_dir_name + "\\" + 'self_n_dir')
            tester_helper.make_dir(net_dir_name + "\\" + 'abv_y_dir')
            tester_helper.make_dir(net_dir_name + "\\" + 'abv_n_dir')
    """
    :arg square_im a binary image of changes in the square
    :return whether there's been a move on the square, below it, or none,
    and the score (if there's a move)
    """

    def get_move(self,sources_place, sources_self, sources_above,
                               targets_place, targets_self, targets_above, tester_info = None):

        try:
            to_save = bool(tester_info)

            if to_save:
                real_move = tester_info[0]
                move_num = tester_info[1]
                angle_idx = tester_info[2]

                self.mistake_idxes = []
                real_change_s = real_move[0]
                real_change_t = real_move[1]
                real_idx_source = sources_place.index(real_change_s)
                real_idx_target = targets_place.index(real_change_t)

            else:
                real_change_s = None
                real_change_t = None
                real_idx_source = None
                real_idx_target = None

            sources_rank = self.check_squares(sources_self,
                                         sources_above,real_change_s, real_idx_source)
            if to_save and SAVE_ALLOT:
                for idx in self.mistake_idxes:
                    tester_helper.save_bw(img=np.array(sources_self[idx]), place=sources_place[idx], move_num=move_num,
                                          angle_idx=angle_idx, desc='dif')
                    tester_helper.save_bw(img=np.array(sources_above[idx]), place=sources_place[idx], move_num=move_num,
                                          angle_idx=angle_idx, desc='dif_abv')


            #save neuron's images:
            cv2.imwrite(self.net_dir_name + "\\" + 'self_y_dir/im'+str(self.neuron_counter)+'.jpg', np.array(sources_self[real_idx_source]))
            cv2.imwrite(self.net_dir_name + "\\" + 'abv_y_dir/im'+str(self.neuron_counter)+'.jpg', np.array(sources_above[real_idx_source]))
            if len(sources_self) > 1:
                n_source_idx = real_idx_source
                while n_source_idx == real_idx_source:
                    n_source_idx = np.random.randint(0, len(sources_self))
                cv2.imwrite(self.net_dir_name + "\\" + 'self_n_dir/im' + str(self.neuron_counter) + '.jpg', np.array(sources_self[n_source_idx]))
                cv2.imwrite(self.net_dir_name + "\\" + 'abv_n_dir/im' + str(self.neuron_counter) + '.jpg', np.array(sources_above[n_source_idx]))

            cv2.imwrite(self.net_dir_name + "\\" + 'self_y_dir/im' + str(self.neuron_counter + 1) + '.jpg', np.array(targets_self[real_idx_target]))
            cv2.imwrite(self.net_dir_name + "\\" +'abv_y_dir/im' + str(self.neuron_counter + 1) + '.jpg', np.array(targets_above[real_idx_target]))
            n_target_idx = real_idx_target
            while n_target_idx == real_idx_target:
                n_target_idx = np.random.randint(0, len(targets_self))
            cv2.imwrite(self.net_dir_name + "\\" + 'self_n_dir/im' + str(self.neuron_counter + 1) + '.jpg', np.array(targets_self[n_target_idx]))
            cv2.imwrite(self.net_dir_name + "\\" + 'abv_n_dir/im' + str(self.neuron_counter + 1) + '.jpg', np.array(targets_above[n_target_idx]))
            self.neuron_counter += 2

            targets_rank = self.check_squares(targets_self,
                                         targets_above,real_change_t, real_idx_target)

            if to_save and SAVE_ALLOT:
                for idx in self.mistake_idxes:
                    tester_helper.save_bw(img=np.array(targets_self[idx]), place=targets_place[idx], move_num=move_num,
                                          angle_idx=angle_idx)
                    tester_helper.save_bw(img=np.array(targets_above[idx]), place=targets_place[idx], move_num=move_num,
                                          angle_idx=angle_idx, desc='abv')
                if(PRINTS):
                    print("sources : ")
                    print(sources_place)
                    print("ranking : ")
                    print(sources_rank)
                    print("dests : ")
                    print(targets_place)
                    print("ranking : ")
                    print(targets_rank)

            return self.get_pairs_and_ranks(sources_place, targets_place, sources_rank,
                                            targets_rank)
        except:
            print("get_move_failed")
            raise
    '''
    receives a list of square images, and list of square images above them and
    returns a list of rank with the corresponding indexes
    '''
    def check_squares(self, squares, above_squares, real_change = None, real_change_idx = None):
        self.mistake_idxes = []
        rank_lst = scikit_learn.check_net(self.net, tester_helper.connect_two_ims_lst(squares, above_squares))
        if real_change is not None:
            for i in range(len(squares)):
                rank = rank_lst[i]
                if real_change is not None:
                    if i == real_change_idx:
                        real_change_ratio = rank
                        self.mistake_idxes.append(i)
            for i in range(len(squares)):
                if rank_lst[i] >= real_change_ratio:
                    self.mistake_idxes.append(i)
            # a metric that consider both the square itself and the square above checks
        return rank_lst

    '''
        receives a sources ranks and places, targets ranks and places, and the chess board, and returns the best pair.
        this is done by find the best match of each source (using best_target_for_given_source method),
        and comparing between all theses matches
        '''

    def get_pairs_and_ranks(self, sources_place, targets_place, source_ranks,
                            target_ranks):
        best_targets_per_source = []
        pairs_rank = []
        pairs = []
        for i in range(len(source_ranks)):
            tmp, best_match_rank = \
                self.best_target_for_given_source(sources_place[i], targets_place, target_ranks)
            best_targets_per_source.append(tmp)
            pairs_rank.append(source_ranks[i] + best_match_rank)
            pairs.append((sources_place[i], tmp))
        return pairs, pairs_rank

    '''
    receives one square of source, all the targets(their places and their ranks), and the chess board and return the
    best target for this source, using the check squares method
    '''
    def best_target_for_given_source(self, source_place, targets_place,
                                     target_ranks):
        target_img_dict = dict(zip(targets_place, target_ranks))
        inv_target_img_dict = dict(zip(target_ranks, targets_place))
        matches = self.chess_helper.square_dests(source_place)  # all the
        # natches of a given square
        best_match_rank = 0
        for match in matches:
            best_match_rank = max(target_img_dict[match], best_match_rank)
        best_match_place = inv_target_img_dict[best_match_rank]

        return best_match_place, best_match_rank






def test_find_move():
    tester = chess_helper.chess_helper(True)
    move_finder = find_moves_rank(tester)
    img = cv2.imread('g1_abv.jpg')

