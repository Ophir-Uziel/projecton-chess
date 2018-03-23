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
if not os.path.exists(IMAGE_FOLDER + "bad/"):
    os.makedirs(os.path.join(IMAGE_FOLDER, 'bad'))
if not os.path.exists(IMAGE_FOLDER + "fixed/"):
    os.makedirs(os.path.join(IMAGE_FOLDER, 'fixed'))
if not os.path.exists(IMAGE_FOLDER + "fixed8/"):
    os.makedirs(os.path.join(IMAGE_FOLDER, 'fixed8'))
if not os.path.exists(IMAGE_FOLDER + "fixer failed/"):
    os.makedirs(os.path.join(IMAGE_FOLDER, 'fixer failed'))

movestr = ""
while True:
    last_error_dir = board_cut_fixer.FixerErrorType.NoDirection.value
    while True:
        try:

            if (direction == connection.LEFT):

                print("please take left photo")
                im = hw.get_image(direction, last_error_dir)
                if last_im_left is not None:
                    fixer.set_prev_im(last_im_left)
            else:
                print("please take right photo")
                im = hw.get_image(direction, last_error_dir)
                if last_im_right is not None:
                    fixer.set_prev_im(last_im_right)

            fix_im = fixer.main(im)
            cv2.imwrite(IMAGE_FOLDER + direction + "_" + str(
                int(img_num)) + '.jpg', im)
            cv2.imwrite(
                IMAGE_FOLDER + "fixed/" + direction + "_" + str(
                    int(img_num))
                + '.jpg', fix_im)
            cv2.imwrite(
                IMAGE_FOLDER + "fixed8/" + direction + "_" + str(
                    int(img_num))
                + '.jpg', fix_im[len(fix_im) // 9:, :])
            # x = input('please accept')
            # if x == "exit":
            #     moves.close()
            #     break
            # if (len(x) != 0):
            #     cv2.imwrite(
            #         IMAGE_FOLDER + "fixer failed/" + direction + "_" + str(
            #             int(img_num))
            #         + '.jpg', im)
            #     raise board_cut_fixer.FixerError("",
            #                                      board_cut_fixer.FixerErrorType.NoDirection)
            if (direction == connection.LEFT):
                last_im_left = fix_im
            else:
                last_im_right = fix_im
            img_num = img_num + 0.5
            break
        except board_cut_fixer.FixerError as e:
            print('bad image, error: ' + str(e.error))
            last_error_dir = e.error.value
            cv2.imwrite(
                IMAGE_FOLDER + "bad/" + direction + "_" + str(
                    int(bad_im_ctr)) +
                '.jpg', im)
            bad_im_ctr += 1
    # flip direction
    # if x=='exit':
    #     break
    if direction == connection.LEFT:
        direction = connection.RIGHT
    else:
        input('exit now')
        movestr += input("Plz enter user move honz") + '\n'
        movestr += input("Plz enter rival move honz")+'\n'
        moves = open(IMAGE_FOLDER + "fixed/moves.txt", 'w+')
        moves.write(movestr)
        moves.close()
        direction = connection.LEFT
