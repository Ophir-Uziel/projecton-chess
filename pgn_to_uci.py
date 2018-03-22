import subprocess
import os


def func(file_name):
    args = ["-W ",file_name]
    proc = subprocess.Popen(args,stdout=subprocess.PIPE)
    while True:
      line = proc.stdout
      if line != '':
        #the real code does filtering here
        print("test:", line.rstrip())
      else:
        break