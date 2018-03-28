from sklearn import svm
import numpy as np
import os
from sklearn import datasets
import cv2
import tester_helper
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
import random
import time
import this
TOP_NET_FILENAME = "top_net_old"
BOTTOM_NET_FILENAME = "bottom_net_old"
NET_FILE = "net2"
PRINT = False

def try1():
    digits = datasets.load_digits()
    print(digits.data)
    print(digits.target)

def rand_img(size):
    img = []
    for i in range(size):
        img.append(np.random.randint(0, 2))
    return img


def create_svm():
    # clf = svm.SVC(gamma=0.000000005)

    clf = svm.SVC(gamma=0.000000001)
    return clf

def create_nets(size,type, nets_num):
    nets = []
    for k in range(nets_num):
        net = MLPClassifier(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(size,), random_state=1, activation= "tanh")
        save_net(net, type + "_net" + "_" + str(k))
        nets.append(net)
    return nets

def train_better_net(nets, imgs_lst_by_class, type):
    i = 0
    carmel_the_gever = 0
    j = 0
    for net in nets:
        shuffle_img, ans = better_shuffle(imgs_lst_by_class)
        print("shuffled")
        inp = np.asarray(shuffle_img)
        tar = np.asarray(ans)
        carmel_the_gever = net.fit(inp, tar)
        print(str(i) + " trained")
        i += 1
        save_net(nets[j],type+'_net_' +str(j))
        j += 1
    return nets, carmel_the_gever
# def train_net(net, y_img, n_img):
#     shuffle_img, ans = shuffle(y_img,n_img)
#     digits = datasets.load_digits()
#     ex_inp = digits.data
#     ex_tsr = digits.target
#     inp = np.asarray(shuffle_img)
#     tar = np.asarray(ans)
#     SVC = net.fit(inp, tar)
#     return net, SVC


def check_net(net, imgs, dir_name = None):
    imgs = im_to_lst(imgs)
    if(dir_name):
        tester_helper.make_dir(dir_name)
        cnt = 0
        for img in imgs:
            if cnt%100 == 0:
                print(str(cnt) + "checked")
            res = net.predict([img])[0]
            img = lst_to_im(img)
            cv2.imwrite(dir_name + "\\" + str(res) + "_" + str(cnt) + ".jpg", np.array(img))
            cnt += 1
    return net.predict(imgs)


def better_shuffle(lsts_lst):
    cntrs = [0]*len(lsts_lst)
    non_full = [i for i in range(len(lsts_lst))]
    shuffle_imgs = []
    ans = []
    total_len = sum([len(lst) for lst in lsts_lst])
    for i in range(total_len):
        for j in range(len(lsts_lst)):
            if len(lsts_lst[j]) == cntrs[j] and j in non_full:
                non_full.remove(j)
        rnd = np.random.choice(non_full)
        shuffle_imgs.append(lsts_lst[rnd][cntrs[rnd]])
        ans.append(rnd)
        cntrs[rnd] += 1
        if i%100==0:
            print("shuffled " + str(i) + " ims")
    return shuffle_imgs, ans



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


def im_to_lst(lst):
    imgs = []
    for i in range(len(lst)):
        if i%100 == 0:
            if(PRINT):
                print("ims to lst: " + str(i))
        new_im =[]
        for row in lst[i]:
            for pixel in list(row):
                new_im.append(pixel)
        imgs.append(new_im)
    return imgs

def read_imgs(lst, fold):
    imgs = []
    to_test = []
    for i in range(len(lst)):
        rnd = np.random.randint(0, 6)
        im = cv2.imread(fold + "\\" +lst[i], cv2.IMREAD_GRAYSCALE)
        new_im =[]
        for row in im:
            for pixel in row:
                new_im.append(pixel)
        if rnd == 0:
            to_test.append(new_im)
        else:
            imgs.append(new_im)
    return imgs, to_test


def lst_to_im(lst):
    img = []
    for i in range(20):
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

def classes_test(classes_folds):
    nets = create_nets()
    lsts_lst, lsts_to_test = [read_imgs(os.listdir("classes\\" + fold), "classes\\" + fold)[0] for fold in classes_folds],\
                             [read_imgs(os.listdir("classes\\" + fold), "classes\\" + fold)[1] for fold in classes_folds]
    train_better_net(net, lsts_lst)
    save_net(net, 'net3')
    results = [check_net(net,lsts_to_test[i], "classes\\" + str(i)) for i in range(len(classes_folds))]
    print(results)
    return net

def get_random_imgs(dir, count ):
    names = random.sample(os.listdir(dir),count)
    ims = [cv2.imread(os.path.join(dir,name),cv2.IMREAD_GRAYSCALE) for name in names]
    return ims

def train_net(type, numpics):
    nets = []
    for j in range(3):
        net = read_net(type + '_net_' +str(j))
        nets.append(net)
    numpics = numpics
    gooddir = "C:\\Users\\t8291043\\Desktop\\good_squares\\to_classify\\"+type+"_good"
    baddir = "C:\\Users\\t8291043\\Desktop\\good_squares\\to_classify\\"+type+"_bad"
    goodims = im_to_lst(get_random_imgs(gooddir, numpics))
    badims = im_to_lst(get_random_imgs(baddir, numpics))
    classes = [goodims, badims]
    print("training!")
    train_better_net(nets, classes, type)


def test_nets(type, name, numpics):
    net = read_net(name)
    numpics = numpics
    gooddir = "C:\\Users\\t8291043\\Desktop\\good_squares\\to_classify\\" + type + "_good"
    baddir = "C:\\Users\\t8291043\\Desktop\\good_squares\\to_classify\\" + type + "_bad"
    goodims = im_to_lst(get_random_imgs(gooddir, numpics))
    badims = im_to_lst(get_random_imgs(baddir, numpics))
    classes_to_test = [goodims, badims]
    results = [check_net(net, classes_to_test[i], "classes\\" + str(i)) for i in range(len(classes_to_test))]
    print("false negative rate:" + str(100*(sum(results[0])/len(results[0])))+"%")
    print("false positive rate:" + str(100 * ((len(results[1])-sum(results[1])) / len(results[1]))) + "%")
    # print(results)
#
#

def night_running(kama_kod):
    create_nets(400, "top", 3)
    create_nets(400, "bottom", 3)

    train_net("top", kama_kod)
    train_net("bottom", kama_kod)

    for i in range(3):
        test_nets("top", "top_net_" + str(i), 5000)
        test_nets("bottom", "bottom_net_" + str(i), 5000)

# gedulim = time.time()
# night_running(40000)
# print((time.time() - gedulim)/3600)
#
#read_test()
# classes = os.listdir("classes")
# classes_test(classes)

