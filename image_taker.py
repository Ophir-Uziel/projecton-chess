import connection
import cv2
import board_cut_fixer
import hardware
import os


IMAGE_FOLDER = "taken photos\\"

""" setup connection """
fixer = board_cut_fixer.board_cut_fixer()
hw = hardware.hardware(2, None)
img_num = 0
direction = connection.LEFT
last_im_right = None
last_im_left = None
bad_im_ctr = 0
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
if not os.path.exists(IMAGE_FOLDER+"bad/"):
    os.makedirs(os.path.join(IMAGE_FOLDER,'bad'))
if not os.path.exists(IMAGE_FOLDER+"fixed/"):
    os.makedirs(os.path.join(IMAGE_FOLDER, 'fixed'))
if not os.path.exists(IMAGE_FOLDER + "fixed8/"):
    os.makedirs(os.path.join(IMAGE_FOLDER, 'fixed8'))
if not os.path.exists(IMAGE_FOLDER+"fixer failed/"):
    os.makedirs(os.path.join(IMAGE_FOLDER, 'fixer failed'))
user_moves = open(IMAGE_FOLDER + "fixed/user_moves.txt",'w+')
rival_moves = open(IMAGE_FOLDER + "fixed/rival_moves.txt",'w+')




while True:

    while True:
        try:

            if (direction == connection.LEFT):

                print("please take left photo")
                im = hw.get_image(direction)
                if last_im_left is not None:
                    fixer.set_prev_im(last_im_left)
            else:
                print("please take right photo")
                im = hw.get_image(direction)
                if last_im_right is not None:
                    fixer.set_prev_im(last_im_right)

            print(len(im))
            fix_im = fixer.main(im)
            cv2.imwrite(IMAGE_FOLDER + direction + "_"+str(int(img_num))+'.jpg',im)
            cv2.imwrite(
                IMAGE_FOLDER +"fixed/"+ direction + "_" + str(int(img_num))
                + '.jpg',fix_im)
            cv2.imwrite(
                IMAGE_FOLDER +"fixed8/"+ direction + "_" + str(int(img_num))
                + '.jpg',fix_im[len(fix_im)//9:,:])
            x = input('please accept')
            if x == "exit":
                user_moves.close()
                rival_moves.close()
                break
            if(len(x)!=0):
                cv2.imwrite(
                    IMAGE_FOLDER + "fixer failed/" + direction + "_" + str(
                        int(img_num))
                    + '.jpg', im)
                raise Exception()
            if(direction==connection.LEFT):
                last_im_left = fix_im
            else:
                last_im_right = fix_im
            img_num = img_num+0.5
            break
        except:
            print('bad image')
            cv2.imwrite(
                IMAGE_FOLDER + "bad/" +direction + "_" + str(int(bad_im_ctr))+
                '.jpg',im)
            bad_im_ctr += 1
    # flip direction

    if direction == connection.LEFT:
        direction = connection.RIGHT
    else:
        user_moves.write(input("Plz enter user move honz"))
        rival_moves.write(input("Plz enter rival move honz"))
        direction = connection.LEFT
    user_moves = open(IMAGE_FOLDER + "fixed/user_moves.txt", 'w+')
    rival_moves = open(IMAGE_FOLDER + "fixed/rival_moves.txt", 'w+')
