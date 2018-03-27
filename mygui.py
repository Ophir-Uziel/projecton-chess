from tkinter import *
import chess
import scipy

from threading import Thread


from scipy import misc
from PIL import Image
from PIL import ImageTk
from threading import Thread
import time
from multiprocessing import Process
import cv2
import copy

###dictionary
DICTIONARY = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7}
IMAGES = []
TEMP_ANGLE_IMGS = []


def add_angle_image(img):
    global TEMP_ANGLE_IMGS
    TEMP_ANGLE_IMGS.append(img)
    print("added")

def add_angle_images():
    global TEMP_ANGLE_IMGS
    global IMAGES
    for img in TEMP_ANGLE_IMGS:
        IMAGES.append(img)
    print(len(IMAGES))
    print("added " + str(len(TEMP_ANGLE_IMGS)) + " images")
    TEMP_ANGLE_IMGS = []

def flush_angle_images():
    global TEMP_ANGLE_IMGS
    TEMP_ANGLE_IMGS = []
    print("flush")

class GameBoard():
    def __init__(self,root,canvas, rows=8, columns=8, size=620,
                 color1="white",
                 color2="#AAAAAF"):
        '''size is the size of a square, in pixels'''
        self.root = root
        self.canvas = canvas
        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.x = self.size*11/14
        self.y = self.size*6/7
        self.pieces = {}
        self.position_of_board = [["r","n","b","q","k","b","n",
                                   "r"],
                                  ["p", "p", "p", "p", "p", "p", "p","p"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["wp","wp","wp","wp","wp","wp","wp","wp"],
                                  ["wr", "wn", "wb", "wq", "wk", "wb", "wn",
                                   "wr"]]

        self.canvas_width = columns * size
        self.canvas_height = rows * size
        self.board_img = self.make_img_from_file("board.png",self.size,self.size)
        self.white_players_turn = self.make_img_from_file(
            "white_players_turn.png",70,300)
        self.black_players_turn = self.make_img_from_file(
            "black_players_turn.png",70,300)
        self.board_state = self.make_img_from_file(
            "board_state.png",50,250)
        self.line_analysis = self.make_img_from_file(
            "line_analysis.png",50,250)
        self.turn = 1

        self.ISSLIDING = False

        self.images_object = [["r","n","b","q","k","b","n","r"],
                                  ["p", "p", "p", "p", "p", "p", "p","p"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["wp","wp","wp","wp","wp","wp","wp","wp"],
                                  ["wr", "wn", "wb", "wq", "wk", "wb", "wn",
                                   "wr"]]

        self.chaneg_player()

    def make_img_from_file(self,file_name,x,y):
        img = Image.open("gui_images\\"+file_name)
        resize_img = Image.fromarray(misc.imresize(img, (x,y)))
        final_img = ImageTk.PhotoImage(resize_img)
        return final_img

    def chaneg_player(self):
        self.turn = self.turn + 1
        if self.turn%2 == 0:
            self.canvas.create_image(600, 40,image =self.white_players_turn)
        elif self.turn%2 == 1:
            self.canvas.create_image(600, 40, image =self.black_players_turn)
        return self.white_players_turn,self.black_players_turn

    def draw_board(self):
        '''draw board'''
        self.canvas.create_image(self.x, self.y, image=self.board_img)
        self.canvas.create_image(self.x, self.y-self.size/2-35,
                                 image=self.board_state)
        #self.canvas.create_image(self.x+100,self.y+100, image = self.r_img)
        return self.board_img #, self.r_img

    def placepiece(self, imageName, row, column):
        '''Place a piece at the given row/column'''
        image = Image.open("gui_images\\"+imageName)
        resize_img = Image.fromarray(misc.imresize(image, (self.size//8
                                                           ,self.size//8)))
        image = ImageTk.PhotoImage(resize_img)
        x0 = int(self.x+((column-8) * self.size//8)) + int(self.size//2)+ \
             int(self.size//16)
        y0 = int(self.y+((row-8) * self.size//8)) + int(self.size//2)+int(self.size//16)
        image_object = self.canvas.create_image( x0, y0+1,image = image)
        return image, image_object

    def draw_position_of_board(self):

        '''draw board and pieces'''
        images = []
        board_img = self.draw_board()
        for i in range(len(self.position_of_board)):
            for j in range(len(self.position_of_board[0])):
                if self.position_of_board[i][j] !="O":
                    image ,image_object =self.placepiece(
                        self.position_of_board[i][
                                              j]+".png",i,j)
                    images.append(image)
                    self.images_object[i][j] = image_object

        return images, board_img

    def make_move(self,move):
        colomn = DICTIONARY[move[0]]
        row = 8-(int(move[1]))
        new_colomn = DICTIONARY[move[2]]
        new_row = 8-(int(move[3]))
        moved_piece = self.position_of_board[row][colomn]
        str_move = move[0]+move[1]

        if len(str_move)==5:        #promotion
            if moved_piece[0]=="w":
                self.position_of_board[new_row][new_colomn] = "w"+str_move[4]
            else:
                self.position_of_board[new_row][new_colomn] = str_move[4]
                self.position_of_board[row][colomn] = "O"
        else:
            if str_move == "e1g1" and moved_piece == "wk":      #castling
                self.position_of_board[new_row][new_colomn] = moved_piece
                moved_piece(("h1","f1"))
            elif str_move == "e1b1" and moved_piece == "wk":
                self.position_of_board[new_row][new_colomn] = moved_piece
                moved_piece(("a1","c1"))

            elif str_move == "e8g8" and moved_piece == "k":
                self.position_of_board[new_row][new_colomn] = moved_piece
                moved_piece(("h8","f8"))
            elif str_move == "e8b8" and moved_piece == "k":
                self.position_of_board[new_row][new_colomn] = moved_piece
                moved_piece(("a8","c8"))
            else:       #regular move
                im5 = self.placepiece("red.png", row, colomn)
                im4 = self.placepiece("red.png", new_row, new_colomn)
                time.sleep(1)
                self.position_of_board[new_row][new_colomn] = moved_piece
                self.position_of_board[row][colomn] = "O"
        image_name = "white.png"
        if (row+colomn)%2==1:
            image_name = "gray.png"
        im3 = self.placepiece(image_name,row,colomn)
        image_name = "white.png"
        if (new_row+new_colomn)%2==1:
            image_name = "gray.png"

        im1 = self.placepiece(image_name,new_row,new_colomn)
        time.sleep(0.5)
        im2 = self.placepiece(str(moved_piece)+".png",new_row,new_colomn)
        return im1, im2, im3, im4, im5

class GIF():
    def __init__(self, x, y,size, root, files_names_lst = None, imageslst =
    None,
                                                                    ):
        self.root = root
        self.x = x
        self.y = y
        self.files_names_lst = files_names_lst
        self.gif_counter = 0
        self.size =size
        self.garbage = []
        if imageslst != None:
            self.images = imageslst
        else:
            self.images = []
        if self.files_names_lst !=None:
            for i in range(len(self.files_names_lst)):
                image = Image.open("gui_images\\"+self.files_names_lst[i])
                resize_img = Image.fromarray(misc.imresize(image,
                                                           (self.size,self.size)))
                final_image = ImageTk.PhotoImage(resize_img)
                self.images.append(final_image)



    def draw_gif(self):
        if self.files_names_lst !=None:
            for i in range(len(self.files_names_lst)):
                image = Image.open("gui_images\\"+self.files_names_lst[i])
                resize_img = Image.fromarray(misc.imresize(image,
                                                           (self.size,self.size)))
                final_image = ImageTk.PhotoImage(resize_img)
                self.images.append(final_image)
        timer_label = Label(self.root, text="")
        timer_label.place(anchor=NW, x=self.x, y=self.y)
        timer_label.configure(image=self.images[self.gif_counter])
        self.gif_counter = (self.gif_counter + 1) % len(self.images)
        x = self.root.after(2000, self.draw_gif)

class CLOCK():
    def __init__(self,root,canvas):
        self.root = root
        self.now = 0
        self.images = []
        self.canvas = canvas
        self.Hight = 70
        self.wight = 45
        for i in range(10):
            num = Image.open("gui_images\\"+str(i)+".png")
            resize_img = Image.fromarray(misc.imresize(num,
                                                       ( self.Hight,self.wight)))
            final_image = ImageTk.PhotoImage(resize_img)
            self.images.append(final_image)

        oo = Image.open("gui_images\\"+"00.png")
        resize_oo = Image.fromarray(misc.imresize(oo,
                                                  (self.Hight,
                                    int(self.Hight/5+1))))
        self.oo = ImageTk.PhotoImage(resize_oo)

    def update_clock(self):
        for img in self.images:
            self.canvas.delete(img)
        self.canvas.create_image(1000-self.wight, self.Hight/2,
                                 image=self.images[
            int(self.now%10)])
        self.canvas.create_image(1000-2*self.wight, self.Hight/2,
                                 image=self.images[
                                     int(self.now/10)%6])
        self.canvas.create_image(1000-3*self.wight+int(self.Hight/5)+2,
                                 self.Hight/2,image =
        self.oo)
        self.canvas.create_image(1000-3*self.wight-int(self.Hight/5),
                                 self.Hight/2,
                                 image=self.images[
            int(self.now/60)%10])
        self.canvas.create_image(1000-4*self.wight-int(self.Hight/5),
                                 self.Hight/2, image=self.images[
                                     int(self.now/600)%10])


        self.now = self.now + 1
        x = self.root.after(990, self.update_clock)

class GUI():


    def __init__(self,gif_images_lsts):
        self.WINDOW_WIDTH = 1250
        self.WINDOW_HEIGHT = 700
        self.root = Tk()
        self.now = 0
        self.root.geometry("1600x1200")
        self.canvas = Canvas(self.root, width=self.WINDOW_WIDTH,
                        height=self.WINDOW_HEIGHT,bg = '#222222')
        self.canvas.pack()
        self.gif_images_lsts=gif_images_lsts
        self.gif_counter = 0

        self.x = 600
        self.y = 200

        #self.label = Label(self.root, )

        self.last_move=""

        self.gif_x = 0
        self.gif_y = 0

        self.realtime_images = []
        self.gif_counter = 0

        self.garbage = []

        self.real_time_gifs = []
        self.board = GameBoard(self.root, self.canvas,size =
        int(self.WINDOW_WIDTH*0.5*(1-1/len(self.gif_images_lsts))))

        self.images_for_realtime_gifs = ['01.png', '001.png']
        self.runtime_counter = 0

        self.thread = None
        self.image_index=0
        self.prev=-1
        self.next=1
        self.last=-0

    def old_update_clock(self):
        timer_label = Label(text="")
        timer_label = Label(self.root, text="", bg="#AFFAAA",
                            fg="orange",justify=RIGHT)
        timer_label.place(anchor=NW, x=self.WINDOW_WIDTH-self.x, y=0)
        timer_label.configure(text=str(self.now // 60) + ':' + str(self.now %
                                                                   60),
                              highlightbackground  = "red",
                              font=("Courier", 44))
        self.now = self.now + 1
        x = self.root.after(1000, self.update_clock)

    def draw_gifs(self):
        gifs = []
        for i in range(len(self.gif_images_lsts)):
            gifs.append( GIF(0,i*self.WINDOW_HEIGHT/len(self.gif_images_lsts),
                          int(self.WINDOW_HEIGHT/len(self.gif_images_lsts))
                         ,self.root,files_names_lst = self.gif_images_lsts[i]))
        for gif in gifs:
            gif.draw_gif()

        return gifs

    def draw_clock(self):
        clock = CLOCK(self.root,self.canvas)
        clock.update_clock()
        return clock

    def draw_image_from_file(self, filename,x,y):
        image = Image.open("gui_images\\"+filename)
        resize_img = Image.fromarray(misc.imresize(image, (300,
                                                           400)))
        final_image = ImageTk.PhotoImage(resize_img)
        self.canvas.create_image(x, y, image=final_image)
        self.garbage.append( final_image)

    def draw_image(self, img,x,y):
        resize_img = Image.fromarray(misc.imresize(img, (350,350)))
        imgcopy = resize_img.copy()
        final_image = ImageTk.PhotoImage(imgcopy)
        self.canvas.create_image(x, y, image=final_image)
        self.garbage.append(imgcopy)
        self.garbage.append(final_image)

        return final_image

    def draw_board(self):
        images = self.board.draw_board()
        images2 = self.board.draw_position_of_board()
        return images, images2

    def make_move(self,move):
        images = self.board.make_move(move)
        self.garbage.append(images)
        self.images_for_realtime_gifs = ['01.png']
        return self.board

    def not_got_image(self):
        return False

    def getImage(self):
        return None

    def changeImage(self, im):
        return False

    def server_wait_image(self):
        '''''
        while True:
            while (self.not_got_image()): pass
            im = self.getImage()
            self.changeImage(im)
        '''''
        return  None

    def set_images_for_real_time(self, files_names_lst):
        for i in range(2):
            for file_name in files_names_lst[i]:
                img = Image.open("gui_images\\"+file_name)
                final_img = ImageTk.PhotoImage(img)
                self.images_for_realtime_gifs[i].append(final_img)

    def draw_next_runtime(self):

        for i in range(2):
            x = self.WINDOW_WIDTH * 7 / 10+int(self.WINDOW_HEIGHT * 7 / 36)
            y = (8 * i + 3) * self.WINDOW_HEIGHT * 1 / 18+int(self.WINDOW_HEIGHT * 7 / 36)

            num_imgs = len(self.images_for_realtime_gifs[i])
            img = self.images_for_realtime_gifs[i][
                                self.runtime_counter % num_imgs]


            self.draw_image(img,x,y)
        self.runtime_counter = self.runtime_counter+1
        self.runner()

    def make_next_button(self):
        img = Image.open("gui_images\\"+"next_image_buttom.png")
        resize_img = Image.fromarray(misc.imresize(img, (65,100)))
        final_img = ImageTk.PhotoImage(resize_img)
        #photo = PhotoImage(file="click.png")  # Give photo an image
        self.garbage.append(final_img)
        b = Button(self.root, image = final_img,bg = "#222222", borderwidth = 0)  # Create a button

        b.configure(command = lambda: self.show_image_by_index(self.next))  #
        # Configure
        # instance
        #  to use
        # the photo
        b.place(relx=11/15, rely=113/140, anchor = CENTER)
        #b.pack()

    def make_prev_button(self,):
        img = Image.open("gui_images\\"+"prev_image_buttom.png")
        resize_img = Image.fromarray(misc.imresize(img, (65,100)))
        final_img = ImageTk.PhotoImage(resize_img)
        #photo = PhotoImage(file="click.png")  # Give photo an image
        self.garbage.append(final_img)
        b = Button(self.root, image = final_img,bg = "#222222", borderwidth = 0)  # Create a button

        b.configure(command = lambda: self.show_image_by_index(self.prev))  #
        # Configure
        # instance
        #  to use
        # the photo
        b.place(relx=8/13, rely=113/140, anchor = CENTER)

    def make_end_button(self,):
        img = Image.open("gui_images\\"+"last_image_buttom.png")
        resize_img = Image.fromarray(misc.imresize(img, (65,100)))
        final_img = ImageTk.PhotoImage(resize_img)
        #photo = PhotoImage(file="click.png")  # Give photo an image
        self.garbage.append(final_img)
        b = Button(self.root, image = final_img,bg = "#222222", borderwidth = 0)  # Create a button

        b.configure(command = lambda: self.show_image_by_index(self.last))  # Configure
        # the earlier
        # instance
        #  to use
        # the photo
        b.place(relx=166/195, rely=113/140, anchor = CENTER)

    def draw_real_gif(self):
        if IMAGES != []:
            for i in range(len(IMAGES)):
                image = Image.open("gui_images\\"+IMAGES[i])
                print(4*self.WINDOW_WIDTH/9,7*self.WINDOW_HEIGHT/18)
                resize_img = Image.fromarray(misc.imresize(image,
                                                           (int(1*self.WINDOW_HEIGHT/2),
                                                            int(7*self.WINDOW_WIDTH/18))))

                final_image = ImageTk.PhotoImage(resize_img)
                self.realtime_images.append(final_image)
            timer_label = Label(self.root, text="")
            timer_label.place(anchor=CENTER, x=int(self.WINDOW_WIDTH*4/5),
                              y=int(self.WINDOW_HEIGHT*1/2)+8)
            timer_label.configure(image=self.realtime_images[self.gif_counter])
            self.gif_counter = (self.gif_counter + 1) % len(self.realtime_images)
        x = self.root.after(2000, self.draw_real_gif)

    def show_image_by_index(self,type):
        if IMAGES != []:
            if type ==  self.prev or type == self.next:
                self.draw_image(IMAGES[self.image_index+type],950,
                                self.WINDOW_HEIGHT/2)
                self.image_index = self.image_index+type
            elif type == self.last:
                self.draw_image(IMAGES[-1],950,self.WINDOW_HEIGHT/2)
                self.image_index = len(IMAGES)

gui = None
def main():
    global gui
    gui = GUI([["our_process.png"], ["one.png", "two.png", "three.png",
                                     "four.png"],
               ["Screenshot_2.png",
                "Screenshot_4.png",
                "Screenshot_5.png",
                "Screenshot_7.png"], ["one1.png",
                                      "three1.png",
                                      "four1.png",
                                      "five1.png"],
               ["R1.png",
                "R2.png", "R3.png", "R4.png"]])

    gui.canvas.pack()
    a = gui.draw_gifs()
    clock = gui.draw_clock()
    d = gui.draw_board()
#    gui.images_for_realtime_gifs=[["one.png", "two.png", "three.png",
#                                     "four.png"],
#              [ "Screenshot_2.png","Screenshot_4.png", "Screenshot_5.png","Screenshot_7.png"]]

#    gui.images_for_realtime_gifs = get_images()

    gui.make_next_button()
    gui.make_prev_button()
    gui.make_end_button()


    gui.root.mainloop()


def make_moves(move):
    time.sleep(1)
    Thread(target=gui.make_move, args=(move,)).start()



'''
    time.sleep(10)
    while True:
        time.sleep(1)
        if check_images():
            gui.images_for_realtime_gifs = get_images()
            gui.runtime_counter = 0



    while True:
        img1  = open("cat.png")
        img2 = open("r.png")
        gui.thread = Thread(target=gui.draw_image,args = (img1, 800,300))
        gui.thread.start()
        time.sleep(5)
        gui.thread = Thread(target=gui.draw_image,args = (img2, 800,300))
        gui.thread.start()


#get_images()
#check_images()
#    lsnr = listener.listener()

#    img1 = lsnr.get_image()
#    img2 = lsnr.get_image
#    cv2.imshow("1",img1)
#    cv2.waitKey(0)

#    gui.draw_image(img1,700,700)

    #t1 = threading.Thread(target=gui.server_wait_image())
#    p1 = Process(target=gui.inon())
#    p1.start()
#    p2 = Process(target=gui.server_wait_image())
#    p2.start()
#    p1.join()
#    p2.join()


    t2 = threading.Thread(target=gui.inon())
    t2.start()
    #t1.start()

    gui.root.after(1000, gui.make_move, ("b3", "e4"))
        '''

'''

img = scipy.misc.imread("0.png", flatten=False, mode="RGBA")
for i in len(img):
    for j in len (img[0]):
        img[i][j][3] = 0.5
scipy.misc.imsave("0", img, format="png")
'''



def init():
    Thread(target = main).start()