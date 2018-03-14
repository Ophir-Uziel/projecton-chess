import socket
import numpy

from threading import Thread

SIZE_LEN = 10

### Error fixing consts ###
REQUEST_SHOT_MSG = "please take photo"
GOOD_SHOT_MSG = "great f*king shot"

LAPTOP_IP = '192.168.43.195'
SEND_TIMEOUT = 2.0 #seconds
LISTEN_TIMEOUT = 1000.0
IM_SIZE = 8192000
SENDER = True
LISTENER = False
PORT = 5000
"""
This class provides sender and receiver TCP services,
sender is nonblocking while receiver obviously is.
"""
class connection:

    def __init__(self, type, port=PORT):

        def setup_socket(me,port, type):
            while True:
                try:
                    if(type == SENDER):
                        me.timeout = SEND_TIMEOUT
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.settimeout(me.timeout)
                        s.connect((LAPTOP_IP, port))
                        print("Connected to GUI!")
                        self.socket = s
                    else:
                        me.timeout = LISTEN_TIMEOUT
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.bind(("", port))
                        self.sock = s
                        self.sock.settimeout(me.timeout)
                        self.sock.listen(1)
                        sender, address = self.sock.accept()
                        print("Successfully connected to pi: ", address)
                        self.socket = sender
                    break
                except:
                    print("Failed to connect to GUI!")

        self.thread = Thread(target=setup_socket, args=(self,port,type))
        self.thread.start()
        self.thread.join()

    def send_msg(self, msg):
        def really_send(msg):
            try:
                self.socket.settimeout(SEND_TIMEOUT)
                self.send_data(msg.encode())
            except:
                print("failed to send msg <:-(")

        if not self.thread is None:
            self.thread.join()

        self.thread = Thread(target=really_send, args=(msg,))
        self.thread.start()

    def get_msg(self):
        self.socket.settimeout(LISTEN_TIMEOUT)
        while True:
            msg = str(self.recv_data())[2:-1]
            if not (msg is None):
               return msg


    def send_data(self, data):
        datalen = str(len(data)).ljust(SIZE_LEN)
        data_final = datalen.encode()+data
        len_sent = 0
        while len_sent<SIZE_LEN:
            l = self.socket.send((data_final[len_sent:SIZE_LEN]))
            len_sent = len_sent + l
            
        while len_sent<len(data_final):
            l = self.socket.send((data_final[len_sent:]))
            len_sent = len_sent + l
        
    def recv_data(self):

        tmp_data = self.socket.recv(IM_SIZE)

        while len(tmp_data)<SIZE_LEN:
            tmp_data = tmp_data + self.socket.recv(IM_SIZE)

        msg_size = int(tmp_data[0:SIZE_LEN].decode())+SIZE_LEN
        while len(tmp_data)<msg_size:
            tmp_data = tmp_data + self.socket.recv(IM_SIZE)
            
        return tmp_data[SIZE_LEN:]        
