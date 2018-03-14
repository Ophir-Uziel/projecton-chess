import connection
import picamera
import RPi.GPIO as gpio
import time 
import cv2

### Debug consts ###
HAS_CAMERA = False

### Game consts ###
NUM_ANGLES = 2


### Hardware consts ###
ul = 21
ur = 5
dr = 13
dl = 4
btn = 17
VibTime = 0.5

def one_still():
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        camera.shutter_speed = camera.exposure_speed
        #    camera.iso=400
        #    camera.exposure_mode='off'
        #    #g=camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = [1.3, 1.3]
        #    camera.color_effects=(64,170)
        #    camera.brightness=55
        #    camera.contrast=20
        #     camera.drc_strength='medium'
        camera.start_preview()
        # Camera warm-up time
        time.sleep(0.1)
        camera.capture('cam.jpg')
        return (cv2.imread('cam.jpg'))

def init_vib():
    gpio.cleanup()
    gpio.setmode(gpio.BCM)
    gpio.setup(ur, gpio.OUT)
    gpio.setup(ul, gpio.OUT)
    gpio.setup(dr, gpio.OUT)
    gpio.setup(dl, gpio.OUT)

def report_bad_cam():
    init_vib()
    gpio.output(ul, gpio.HIGH)
    gpio.output(dl, gpio.HIGH)
    gpio.output(ur, gpio.HIGH)
    gpio.output(dr, gpio.HIGH)
    time.sleep(VibTime/2)
    gpio.output(ul, gpio.LOW)
    gpio.output(dl, gpio.LOW)
    gpio.output(ur, gpio.LOW)
    gpio.output(dr, gpio.LOW)
    time.sleep(VibTime/2)


def request_shot(shot_dir):
    init_vib()
    if(shot_dir == connection.RIGHT):
        gpio.output(ur, gpio.HIGH)
        gpio.output(dr, gpio.HIGH)
        time.sleep(VibTime/2)
        gpio.output(ur, gpio.LOW)
        gpio.output(dr, gpio.LOW)
        time.sleep(VibTime/2)
    elif(shot_dir == connection.LEFT):
        gpio.output(ul, gpio.HIGH)
        gpio.output(dl, gpio.HIGH)
        time.sleep(VibTime/2)
        gpio.output(ul, gpio.LOW)
        gpio.output(dl, gpio.LOW)
        time.sleep(VibTime/2)

def report_good_shot():
    init_vib()
    give_vibration(ur, vibtime=VibTime/2)
    give_vibration(ul,vibtime=VibTime/2)
    give_vibration(dl,vibtime=VibTime/2)
    give_vibration(dr,vibtime=VibTime/2)

def take_shot(shot_dir):
    if(HAS_CAMERA):
        while True:
            request_shot(shot_dir)
            gpio.cleanup()
            gpio.setmode(gpio.BOARD)
            gpio.setup(btn, gpio.IN, pull_up_down=gpio.PUD_DOWN)
            try:
                print("plz press me honz")
                
                while True:
                    if (gpio.input(btn) == 1):
                        break

                img = one_still()
                # img = cv2.resize(img,(RESIZE_SIZE,RESIZE_SIZE))
                should_enter = False
                gpio.cleanup()
                break

            except KeyboardInterrupt:
                print('cam err!')
                report_bad_cam()
                
    else:
        request_shot(shot_dir)
        img = cv2.imread('0.jpg')    
    con.send_image(img) #non-blocking


def give_vibration(pin, vibtime=VibTime):
    gpio.output(pin, gpio.HIGH)
    time.sleep(vibtime)
    gpio.output(pin, gpio.LOW)
    time.sleep(vibtime)

def indicate_move(move):

    init_vib()

    move = [move[0:2], move[2:4]]

    for string in move:
        if (string == "a1"):
            give_vibration(dl)
            give_vibration(dl)
            give_vibration(dl)

        elif (string == "a2"):
            give_vibration(dl)
            give_vibration(dl)
            give_vibration(ul)

        elif (string == "a3"):
            give_vibration(dl)
            give_vibration(ul)
            give_vibration(dl)

        elif (string == "a4"):
            give_vibration(dl)
            give_vibration(ul)
            give_vibration(ul)

        elif (string == "a5"):
            give_vibration(ul)
            give_vibration(dl)
            give_vibration(dl)

        elif (string == "a6"):
            give_vibration(ul)
            give_vibration(dl)
            give_vibration(ul)

        elif (string == "a7"):
            give_vibration(ul)
            give_vibration(ul)
            give_vibration(dl)

        elif (string == "a8"):
            give_vibration(ul)
            give_vibration(ul)
            give_vibration(ul)

        elif (string == "b1"):
            give_vibration(dl)
            give_vibration(dl)
            give_vibration(dr)

        elif (string == "b2"):
            give_vibration(dl)
            give_vibration(dl)
            give_vibration(ur)

        elif (string == "b3"):
            give_vibration(dl)
            give_vibration(ul)
            give_vibration(dr)

        elif (string == "b4"):
            give_vibration(dl)
            give_vibration(ul)
            give_vibration(ur)

        elif (string == "b5"):
            give_vibration(ul)
            give_vibration(dl)
            give_vibration(dr)

        elif (string == "b6"):
            give_vibration(ul)
            give_vibration(dl)
            give_vibration(ur)

        elif (string == "b7"):
            give_vibration(ul)
            give_vibration(ul)
            give_vibration(dr)

        elif (string == "b8"):
            give_vibration(ul)
            give_vibration(ul)
            give_vibration(ur)

        elif (string == "c1"):
            give_vibration(dl)
            give_vibration(dr)
            give_vibration(dl)

        elif (string == "c2"):
            give_vibration(dl)
            give_vibration(dr)
            give_vibration(ul)

        elif (string == "c3"):
            give_vibration(dl)
            give_vibration(ur)
            give_vibration(dl)

        elif (string == "c4"):
            give_vibration(dl)
            give_vibration(ur)
            give_vibration(ul)

        elif (string == "c5"):
            give_vibration(ul)
            give_vibration(dr)
            give_vibration(dl)

        elif (string == "c6"):
            give_vibration(ul)
            give_vibration(dr)
            give_vibration(ul)

        elif (string == "c7"):
            give_vibration(ul)
            give_vibration(ur)
            give_vibration(dl)

        elif (string == "c8"):
            give_vibration(ul)
            give_vibration(ur)
            give_vibration(ul)

        elif (string == "d1"):
            give_vibration(dl)
            give_vibration(dr)
            give_vibration(dr)

        elif (string == "d2"):
            give_vibration(dl)
            give_vibration(dr)
            give_vibration(ur)

        elif (string == "d3"):
            give_vibration(dl)
            give_vibration(ur)
            give_vibration(dr)

        elif (string == "d4"):
            give_vibration(dl)
            give_vibration(ur)
            give_vibration(ur)

        elif (string == "d5"):
            give_vibration(ul)
            give_vibration(dr)
            give_vibration(dr)

        elif (string == "d6"):
            give_vibration(ul)
            give_vibration(dr)
            give_vibration(ur)

        elif (string == "d7"):
            give_vibration(ul)
            give_vibration(ur)
            give_vibration(dr)

        elif (string == "d8"):
            give_vibration(ul)
            give_vibration(ur)
            give_vibration(ur)

        elif (string == "e1"):
            give_vibration(dr)
            give_vibration(dl)
            give_vibration(dl)

        elif (string == "e2"):
            give_vibration(dr)
            give_vibration(dl)
            give_vibration(ul)

        elif (string == "e3"):
            give_vibration(dr)
            give_vibration(ul)
            give_vibration(dl)

        elif (string == "e4"):
            give_vibration(dr)
            give_vibration(ul)
            give_vibration(ul)

        elif (string == "e5"):
            give_vibration(ur)
            give_vibration(dl)
            give_vibration(dl)

        elif (string == "e6"):
            give_vibration(ur)
            give_vibration(dl)
            give_vibration(ul)

        elif (string == "e7"):
            give_vibration(ur)
            give_vibration(ul)
            give_vibration(dl)

        elif (string == "e8"):
            give_vibration(ur)
            give_vibration(ul)
            give_vibration(ul)

        elif (string == "f1"):
            give_vibration(dr)
            give_vibration(dl)
            give_vibration(dr)

        elif (string == "f2"):
            give_vibration(dr)
            give_vibration(dl)
            give_vibration(ur)

        elif (string == "f3"):
            give_vibration(dr)
            give_vibration(ul)
            give_vibration(dr)

        elif (string == "f4"):
            give_vibration(dr)
            give_vibration(ul)
            give_vibration(ur)

        elif (string == "f5"):
            give_vibration(ur)
            give_vibration(dl)
            give_vibration(dr)

        elif (string == "f6"):
            give_vibration(ur)
            give_vibration(dl)
            give_vibration(ur)

        elif (string == "f7"):
            give_vibration(ur)
            give_vibration(ul)
            give_vibration(dr)

        elif (string == "f8"):
            give_vibration(ur)
            give_vibration(ul)
            give_vibration(ur)

        elif (string == "g1"):
            give_vibration(dr)
            give_vibration(dr)
            give_vibration(dl)

        elif (string == "g2"):
            give_vibration(dr)
            give_vibration(dr)
            give_vibration(ul)

        elif (string == "g3"):
            give_vibration(dr)
            give_vibration(ur)
            give_vibration(dl)

        elif (string == "g4"):
            give_vibration(dr)
            give_vibration(ur)
            give_vibration(ul)

        elif (string == "g5"):
            give_vibration(ur)
            give_vibration(dr)
            give_vibration(dl)


        elif (string == "g6"):
            give_vibration(ur)
            give_vibration(dr)
            give_vibration(ul)

        elif (string == "g7"):
            give_vibration(ur)
            give_vibration(ur)
            give_vibration(dl)

        elif (string == "g8"):
            give_vibration(ur)
            give_vibration(ur)
            give_vibration(ul)

        elif (string == "h1"):
            give_vibration(dr)
            give_vibration(dr)
            give_vibration(dr)

        elif (string == "h2"):
            give_vibration(dr)
            give_vibration(dr)
            give_vibration(ur)

        elif (string == "h3"):
            give_vibration(dr)
            give_vibration(ur)
            give_vibration(dr)

        elif (string == "h4"):
            give_vibration(dr)
            give_vibration(ur)
            give_vibration(ur)

        elif (string == "h5"):
            give_vibration(ur)
            give_vibration(dr)
            give_vibration(dr)

        elif (string == "h6"):
            give_vibration(ur)
            give_vibration(dr)
            give_vibration(ur)

        elif (string == "h7"):
            give_vibration(ur)
            give_vibration(ur)
            give_vibration(dr)

        elif (string == "h8"):
            give_vibration(ur)
            give_vibration(ur)
            give_vibration(ur)
        if string == move[0]:
            time.sleep(1.5)

    gpio.cleanup()


"""  Init  """
con = connection.connection(connection.SENDER)

""" Boot test """
# vib test
report_good_shot()
report_good_shot()

""" Game loop """
while True:
    msg = con.get_msg()
    print('msg is '+msg)
    if msg[0:-1] == connection.REQUEST_SHOT_MSG:
        take_shot(msg[-1])
    elif msg[0:-4] == connection.MOVE_MSG:
        indicate_move(msg[-4:])
