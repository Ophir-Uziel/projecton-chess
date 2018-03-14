import time
import connection


"""  Init  """
con = connection.connection(connection.SENDER)

""" Boot test """
# vib test

""" Game loop """
while True:
    msg = con.get_msg()
    print('msg is '+msg)
    print('size is'+str(len(msg)))
    con.send_msg(input('input msg'))
