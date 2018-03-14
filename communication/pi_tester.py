import connection
import cv2

""" setup connection """
con = connection.connection(connection.LISTENER)

while True:
    x = input('1 for img, 2 to say good shot, else input a move')
    if x=='1':
        con.send_msg(connection.REQUEST_SHOT_MSG)
        im = con.get_image()
        cv2.imshow("received image",im)
        cv2.waitKey(0)
    elif x=='2':
        con.send_msg(connection.GOOD_SHOT_MSG)
    else:
        con.send_msg(x)
