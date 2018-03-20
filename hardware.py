import os
import cv2
import time
from scipy import misc
import gui_img_manager
import connection
"""
This file is for user communication and hardware.
"""

save_and_print = True
RESIZE_SIZE = 600

class hardware:

    def __init__(self, angle_num, imgs_if_tester):
        if imgs_if_tester is not None: 
            self.is_test = True
            self.angles_imgs_lst = []
            self.angles_imgs_counter = []
            for i in range(angle_num):
                img_names = os.listdir(imgs_if_tester[i])
                sorted_img_names = sorted(img_names, key= first_2_chars)
                img_array = []
                for j in range(len(sorted_img_names)):
                    img_array.append(cv2.imread(imgs_if_tester[i] +'/'+
                                              sorted_img_names[j], cv2.IMREAD_COLOR))

                print(sorted_img_names)
                self.angles_imgs_lst.append(img_array)
                self.angles_imgs_counter.append(0)
        else:
            self.is_test = False
            self.socket = connection.connection(connection.LISTENER)

    def get_image(self, angle_idx):
        if self.is_test:
            img = self.angles_imgs_lst[angle_idx][self.angles_imgs_counter[angle_idx]]
            #img = cv2.resize(img,(RESIZE_SIZE,RESIZE_SIZE))
            gui_img_manager.add_img(img)
            self.angles_imgs_counter[angle_idx] += 1

            return img
        else:
            if(angle_idx==0):
                angle = connection.RIGHT
            else:
                angle = connection.LEFT
            self.socket.send_msg(connection.REQUEST_SHOT_MSG + angle)
            img =self.socket.get_image()
            gui_img_manager.add_img(img)
            return img



    def is_i_first(self):
        return True
    # TODO write this func

    def player_indication(self, move):
            return 0

def first_2_chars(x):
    return x[0:-4]


