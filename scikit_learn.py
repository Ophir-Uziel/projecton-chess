from sklearn import svm
import numpy as np
import os
from sklearn import datasets
import cv2
import tester_helper
from sklearn.externals import joblib


NET_FILE = "net"


def try1():
    digits = datasets.load_digits()
    print(digits.data)
    print(digits.target)

def rand_img(size):
    img = []
    for i in range(size):
        img.append(np.random.randint(0, 2))
    return img


def create_net():
    clf = svm.SVC(gamma=0.000000005)
    return clf


def train_net(net, y_img, n_img):
    shuffle_img, ans = shuffle(y_img,n_img)
    digits = datasets.load_digits()
    ex_inp = digits.data
    ex_tsr = digits.target
    inp = np.asarray(shuffle_img)
    tar = np.asarray(ans)
    SVC = net.fit(inp, tar)
    return net, SVC


def check_net(net, imgs, dir_name = None):
    if(dir_name):
        tester_helper.make_dir(dir_name)
        cnt = 0
        for img in imgs:
            if cnt%10 == 0:
                print(cnt)
            res = net.predict([img])[0]
            img = lst_to_im(img)
            cv2.imwrite(dir_name + "\\" + str(res) + "_" + str(cnt) + ".jpg", np.array(img))
            cnt += 1
    return net.predict(imgs)


def shuffle(arr1,arr2):
    cnt1 = 0
    cnt2 = 0
    shuffle_imgs = []
    ans = []
    for i in range(len(arr1)+len(arr2)):
        if cnt1 == len(arr1):
            shuffle_imgs.append(arr2[cnt2])
            cnt2+=1
            ans.append(0)
        elif cnt2 == len(arr2):
            shuffle_imgs.append(arr1[cnt1])
            cnt1+=1
            ans.append(1)
        else:
            rnd = np.random.randint(0,2)
            if rnd == 0:
                shuffle_imgs.append(arr1[cnt1])
                cnt1 += 1
                ans.append(1)
            else:
                shuffle_imgs.append(arr2[cnt2])
                cnt2 += 1
                ans.append(0)
    return shuffle_imgs, ans


def test(size,im_num):
    net = create_net()
    y_img = []
    n_img = []
    for i in range(im_num):
        img = rand_img(size)
        if i < im_num//2:
            y_img.append(img)
        else:
            n_img.append(img)
    net, SVC = train_net(net,y_img,n_img)
    out = check_net(net, n_img)
    print(out)


def read_imgs(lst, fold):
    imgs = []
    for i in range(len(lst)):
        im = cv2.imread(fold + "\\" +lst[i], cv2.IMREAD_GRAYSCALE)
        new_im =[]
        for row in im:
            for pixel in row:
                new_im.append(pixel)
        imgs.append(new_im)
    return imgs


def lst_to_im(lst):
    img = []
    for i in range(40):
        row =[]
        img.append(row)
        for j in range(20):
            row.append(lst[20*i+j])
    return img

def save_net(net, filename):
    joblib.dump(net, filename+'.pkl')
    return True

def read_net(filename):
    net = joblib.load(filename + '.pkl')
    return net

def test2():
    net = create_net()
    y_lst = read_imgs(os.listdir("y"),"y")
    n_lst = read_imgs(os.listdir("n"),"n")
    y_check = read_imgs(os.listdir("ycheck"),"ycheck")
    n_check = read_imgs(os.listdir("ncheck"),"ncheck")
    train_net(net,y_lst, n_lst)
    save_net(net, 'net')
    n_results = check_net(net, n_check, "n_results")
    print("n_result")
    print(n_results)
    print(str(sum(n_results)) + " out of " + str(len(n_results)))

    y_results = check_net(net, y_check, "y_results")
    print("y_result")
    print(y_results)
    print(str(sum(y_results)) + " out of " + str(len(y_results)))


def read_test():
    y_check = read_imgs(os.listdir("ycheck"), "ycheck")
    n_check = read_imgs(os.listdir("ncheck"), "ncheck")
    net = read_net("net")
    n_results = check_net(net, n_check, "n_results")
    print("n_result")
    print(n_results)
    print(str(sum(n_results)) + " out of " + str(len(n_results)))

    y_results = check_net(net, y_check, "y_results")
    print("y_result")
    print(y_results)
    print(str(sum(y_results)) + " out of " + str(len(y_results)))


read_test()


