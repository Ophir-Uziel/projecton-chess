import subprocess
import os
import tester_helper

def get_moves_from_pgn_file(file_name, moves_file_name):
    args = ["pgn-extract", "-Wuci", file_name]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    while True:
        line = proc.stdout.readline()
        line = line.decode("utf-8")
        if len(line) > 0 and ord("a") <= ord(line[0]) <= ord("h"):
            break
    moves_file = open(moves_file_name, 'w')
    moves = line.split(" ")[:-1]
    for move in moves:
        moves_file.write(move + "\n")
    moves_file.close()
    return moves

def get_moves_files(dir):
    tester_helper.make_dir("move_files")
    files_names = os.listdir(dir)
    for i in range(len(files_names)):
        get_moves_from_pgn_file(dir + "\\" + files_names[i], "move_files\\moves" + str(i))


get_moves_files("pgn_games")
