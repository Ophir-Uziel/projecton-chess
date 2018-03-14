import connection
import cv2

IMAGE_FOLDER = "taken photos/"

""" setup connection """
con = connection.connection(connection.LISTENER)

img_num = 0
direction = connection.LEFT

while True:
    x = input('nothing for img, 2 to say good shot, else input a move')
    if x=='':
        con.send_msg(connection.REQUEST_SHOT_MSG+direction)
        if(direction == connection.LEFT):
            print("please take left photo")
            direction = connection.RIGHT
        else:
            print("please take right photo")
            direction = connection.LEFT
        im = con.get_image()
        print(IMAGE_FOLDER + str(img_num)+'.jpg')
        cv2.imwrite(IMAGE_FOLDER + direction + str(img_num)+'.jpg',im)
        img_num = img_num+1
    elif x=='2':
        con.send_msg(connection.GOOD_SHOT_MSG)
    else:
        con.send_msg(x)
