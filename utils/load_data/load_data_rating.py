#coding=utf-8
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def outfile(file,outstr):
    with open(file, 'awb+') as f:
        f.write(outstr +"\n")

def load_data_rating(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'category'],
                     test_size=0.1, sep="\t"):
    '''
    Loading the data for rating prediction task
    :param path: the path of the dataset, datasets should be in the CSV format
    :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
    :param test_size: the test ratio, default 0.1
    :param sep: the seperator for csv colunms, defalut space
    :return:
    '''

    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    if path.find("ratings_t.dat") > 0:
        n_items = 3952
    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_matrix.todok(), n_users, n_items

def load_data_rating_menu(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'category'],
                     test_size=0.1, sep="\t"):
    '''
    Loading the data for rating prediction task
    :param path: the path of the dataset, datasets should be in the CSV format
    :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
    :param test_size: the test ratio, default 0.1
    :param sep: the seperator for csv colunms, defalut space
    :return:
    '''
    #dictionary
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    n_dire = 0
    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_user = []
    train_movie = []
    train_dire = []
    train_rating = []
    train_data_dir = {}

    for line in train_data.itertuples():
        u = line[1] - 1
        if n_users < u:
            n_users = u
        i = line[2] - 1
        if n_items < i:
            n_items = i
        dire = line[6]
        train_user.append(u)
        train_movie.append(i)
        train_dire.append(dire)
        train_rating.append(line[3])
        train_data_dir[(u,i,dire)] = line[3]
        if line[6] > n_dire:
            n_dire = line[6] + 1


    test_row = []
    test_col = []
    test_rating = []
    test_dire = []
    test_data_dir = {}
    for line in test_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        if n_users < u:
            n_users = u
        i = line[2] - 1
        if n_items < i:
            n_items = i
        dire = line[6]
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
        test_dire.append(line[6])
        test_data_dir[(u,i,dire)] = line[3]
        if line[6] > n_dire:
            n_dire = line[6] + 1
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_data_dir, test_data_dir, (n_users + 1), (n_items + 1) ,n_dire + 1

def load_data_rating_dir(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'category'],
                     test_size=0.1, sep="\t"):
    '''
    Loading the data for rating prediction task
    :param path: the path of the dataset, datasets should be in the CSV format
    :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
    :param test_size: the test ratio, default 0.1
    :param sep: the seperator for csv colunms, defalut space
    :return:
    '''
    #dictionary
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    n_dire = 0
    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_user = []
    train_movie = []
    train_dire = []
    train_rating = []
    train_data_dir = {}
    #将所有只看了一个导演的电影 固定一个参数 其余单独计算
    #一个字典，存储所有用户-导演对的信息，value是这一个对的编号。他产生一个bias,一个特殊的是 为None的时候，是对应一类只看过一个的电影
    user_item_bias_index = {}
    #开始是1
    bias_index = 1
    for line in train_data.itertuples():
        u = line[1] - 1
        if n_users < u:
            n_users = u
        i = line[2] - 1
        if n_items < i:
            n_items = i
        dire = 0
        if line[6] == 1:
            if (u,None) not in user_item_bias_index.keys():
                user_item_bias_index[(u, None)] = bias_index
                bias_index += 1
            dire = user_item_bias_index[(u, None)]
        else:
            if (u,line[7]) not in user_item_bias_index.keys():
                user_item_bias_index[(u, line[7])] = bias_index
                bias_index += 1
            dire = user_item_bias_index[(u, line[7])]
        train_user.append(u)
        train_movie.append(i)
        train_dire.append(dire)
        train_rating.append(line[3])
        train_data_dir[(u,i,dire)] = line[3]



    test_row = []
    test_col = []
    test_rating = []
    test_dire = []
    test_data_dir = {}
    for line in test_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        if n_users < u:
            n_users = u
        i = line[2] - 1
        if n_items < i:
            n_items = i
        dire = 0
        if line[6] == 1:
            if (u, None) not in user_item_bias_index.keys():
                user_item_bias_index[(u, None)] = bias_index
                bias_index += 1
            dire = user_item_bias_index[(u, None)]
        else:
            if (u, line[7]) not in user_item_bias_index.keys():
                user_item_bias_index[(u, line[7])] = bias_index
                bias_index += 1
            dire = user_item_bias_index[(u, line[7])]
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(line[3])
        test_dire.append(line[6])
        test_data_dir[(u,i,dire)] = line[3]
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items,"Number of dire num:",bias_index)
    return train_data_dir, test_data_dir, (n_users + 1), (n_items + 1) ,bias_index
#1	1270	5	978300055	1	1	709	Robert Zemeckis	1	0	5
def load_data_rating_menu_dire_neg_pos(trainpath="../data/ml1m/train_1m_ratings.dat", testpath="../data/ml1m/test_1m_ratings.dat",\
                                       header=['user_id', 'item_id', 'rating', 'timestamp','dire_thistime','dire_allnum',\
                                               'dire_index','dire_name','pos','neg','scoreseq'],
                     test_size=0.1, sep="\t"):
    '''
    Loading the data for rating prediction task
    :param path: the path of the dataset, datasets should be in the CSV format
    :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
    :param test_size: the test ratio, default 0.1
    :param sep: the seperator for csv colunms, defalut space
    :return:
    '''    #dictionary
    #获取用户和电影个数
    user_list = []
    movie_list = []
    dire_num = []
    n_users = 0
    n_items = 0
    n_dire = 0
    with open(trainpath,'r') as f_train:
        for line in f_train:
            line_split = line.strip().split("\t")
            user_list.append(int(line_split[0]))
            movie_list.append(int(line_split[1]))
            dire_num.append(int(line_split[8]))
            dire_num.append(int(line_split[9]))
            if n_users < int(line_split[0]):
                n_users = int(line_split[0])
            if n_items < int(line_split[1]):
                n_items = int(line_split[1])
            if n_dire < int(line_split[8]):
                n_dire = int(line_split[8])
            if n_dire < int(line_split[9]):
                n_dire = int(line_split[9])
    with open(testpath, 'r') as f_test:
        for line in f_test:
            line_split = line.strip().split("\t")
            user_list.append(int(line_split[0]))
            movie_list.append(int(line_split[1]))
            dire_num.append(int(line_split[8]))
            dire_num.append(int(line_split[9]))
            if n_users < int(line_split[0]):
                n_users = int(line_split[0])
            if n_items < int(line_split[1]):
                n_items = int(line_split[1])
            if n_dire < int(line_split[8]):
                n_dire = int(line_split[8])
            if n_dire < int(line_split[9]):
                n_dire = int(line_split[9])
    dftrain = pd.read_csv(trainpath, sep=sep, names=header, engine='python')
    dftest = pd.read_csv(testpath, sep=sep, names=header, engine='python')
    train_data = pd.DataFrame(dftrain)
    test_data = pd.DataFrame(dftest)

    train_user = []
    train_movie = []
    train_dire_pos = []
    train_dire_neg = []
    train_rating = []
    train_all_dire_num = []
    train_data_list = []
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        dire_pos = line[9]
        dire_neg = line[10]
        train_user.append(u)
        train_movie.append(i)
        train_rating.append(line[3])
        train_dire_pos.append(dire_pos)
        train_dire_neg.append(dire_neg)
        # train_dire_pos.append(0)
        # train_dire_neg.append(0)
    train_data_list.append(train_user)
    train_data_list.append(train_movie)
    train_data_list.append(train_rating)
    train_data_list.append(train_dire_pos)
    train_data_list.append(train_dire_neg)


    test_user = []
    test_moivie = []
    test_rating = []
    test_dire_pos = []
    test_dire_neg = []
    test_all_dire_num = []
    test_data_list = []
    for line in test_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        dire_pos = line[9]
        dire_neg = line[10]
        test_user.append(u)
        test_moivie.append(i)
        test_rating.append(line[3])
        test_dire_pos.append(dire_pos)
        test_dire_neg.append(dire_neg)
        # test_dire_pos.append(0)
        # test_dire_neg.append(0)
    test_data_list.append(test_user)
    test_data_list.append(test_moivie)
    test_data_list.append(test_rating)
    test_data_list.append(test_dire_pos)
    test_data_list.append(test_dire_neg)
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_data_list, test_data_list, (n_users + 1), (n_items + 1) ,n_dire + 1

def load_data_rating_fm_dire_neg_pos(trainpath="../data/ml1m/train_1m_ratings.dat", testpath="../data/ml1m/test_1m_ratings.dat",\
                                       header=['user_id', 'item_id', 'rating', 'timestamp','dire_thistime','dire_allnum',\
                                               'dire_index','dire_name','pos','neg','scoreseq'],
                     test_size=0.1, sep="\t"):
    '''
    Loading the data for rating prediction task
    :param path: the path of the dataset, datasets should be in the CSV format
    :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
    :param test_size: the test ratio, default 0.1
    :param sep: the seperator for csv colunms, defalut space
    :return:
    '''    #dictionary
    #获取用户和电影个数
    user_list = []
    movie_list = []
    dire_num = []
    n_users = 0
    n_items = 0
    n_dire = 0
    with open(trainpath,'r') as f_train:
        for line in f_train:
            line_split = line.strip().split("\t")
            user_list.append(int(line_split[0]))
            movie_list.append(int(line_split[1]))
            dire_num.append(int(line_split[8]))
            dire_num.append(int(line_split[9]))
            if n_users < int(line_split[0]):
                n_users = int(line_split[0])
            if n_items < int(line_split[1]):
                n_items = int(line_split[1])
            if n_dire < int(line_split[8]):
                n_dire = int(line_split[8])
            if n_dire < int(line_split[9]):
                n_dire = int(line_split[9])
    with open(testpath, 'r') as f_test:
        for line in f_test:
            line_split = line.strip().split("\t")
            user_list.append(int(line_split[0]))
            movie_list.append(int(line_split[1]))
            dire_num.append(int(line_split[8]))
            dire_num.append(int(line_split[9]))
            if n_users < int(line_split[0]):
                n_users = int(line_split[0])
            if n_items < int(line_split[1]):
                n_items = int(line_split[1])
            if n_dire < int(line_split[8]):
                n_dire = int(line_split[8])
            if n_dire < int(line_split[9]):
                n_dire = int(line_split[9])
    dftrain = pd.read_csv(trainpath, sep=sep, names=header, engine='python')
    dftest = pd.read_csv(testpath, sep=sep, names=header, engine='python')
    train_data = pd.DataFrame(dftrain)
    test_data = pd.DataFrame(dftest)

    train_user = []
    train_movie = []
    train_dire_pos = []
    train_dire_neg = []
    train_rating = []
    train_all_dire_num = []
    train_data_list = []
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        dire_pos = line[9]
        dire_neg = line[10]
        train_user.append(u)
        train_movie.append(i)
        train_rating.append(line[3])
        train_dire_pos.append(dire_pos)
        train_dire_neg.append(dire_neg)
        # train_dire_pos.append(0)
        # train_dire_neg.append(0)
    train_data_list.append(train_user)
    train_data_list.append(train_movie)
    train_data_list.append(train_rating)
    train_data_list.append(train_dire_pos)
    train_data_list.append(train_dire_neg)


    test_user = []
    test_moivie = []
    test_rating = []
    test_dire_pos = []
    test_dire_neg = []
    test_all_dire_num = []
    test_data_list = []
    for line in test_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        dire_pos = line[9]
        dire_neg = line[10]
        test_user.append(u)
        test_moivie.append(i)
        test_rating.append(line[3])
        test_dire_pos.append(dire_pos)
        test_dire_neg.append(dire_neg)
        # test_dire_pos.append(0)
        # test_dire_neg.append(0)
    test_data_list.append(test_user)
    test_data_list.append(test_moivie)
    test_data_list.append(test_rating)
    test_data_list.append(test_dire_pos)
    test_data_list.append(test_dire_neg)
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_data_list, test_data_list, (n_users + 1), (n_items + 1) ,n_dire + 1

#1	1270	5	978300055	1	1	709	Robert Zemeckis	1	0	5
def load_data_rating_menu_dire_neg_pos_added(trainpath="../data/ml1m/train_1m_ratings.dat", testpath="../data/ml1m/test_1m_ratings.dat",\
                                       header=['user_id', 'item_id', 'rating', 'timestamp','dire_thistime','dire_allnum',\
                                               'dire_index','dire_name','pos','neg','scoreseq'],
                     test_size=0.1, sep="\t"):
    '''
    Loading the data for rating prediction task
    :param path: the path of the dataset, datasets should be in the CSV format
    :param header: the header of the CSV format, the first three should be: user_id, item_id, rating
    :param test_size: the test ratio, default 0.1
    :param sep: the seperator for csv colunms, defalut space
    :return:
    '''    #dictionary
    #获取用户和电影个数
    user_list = []
    movie_list = []
    dire_num = []
    n_users = 0
    n_items = 0
    n_dire = 0
    with open(trainpath,'r') as f_train:
        for line in f_train:
            line_split = line.strip().split("\t")
            user_list.append(int(line_split[0]))
            movie_list.append(int(line_split[1]))
            dire_num.append(int(line_split[8]))
            dire_num.append(int(line_split[9]))
            if n_users < int(line_split[0]):
                n_users = int(line_split[0])
            if n_items < int(line_split[1]):
                n_items = int(line_split[1])
            if n_dire < int(line_split[8]):
                n_dire = int(line_split[8])
            if n_dire < int(line_split[9]):
                n_dire = int(line_split[9])
    with open(testpath, 'r') as f_test:
        for line in f_test:
            line_split = line.strip().split("\t")
            user_list.append(int(line_split[0]))
            movie_list.append(int(line_split[1]))
            dire_num.append(int(line_split[8]))
            dire_num.append(int(line_split[9]))
            if n_users < int(line_split[0]):
                n_users = int(line_split[0])
            if n_items < int(line_split[1]):
                n_items = int(line_split[1])
            if n_dire < int(line_split[8]):
                n_dire = int(line_split[8])
            if n_dire < int(line_split[9]):
                n_dire = int(line_split[9])
    dftrain = pd.read_csv(trainpath, sep=sep, names=header, engine='python')
    dftest = pd.read_csv(testpath, sep=sep, names=header, engine='python')
    train_data = pd.DataFrame(dftrain)
    test_data = pd.DataFrame(dftest)

    train_user = []
    train_movie = []
    train_dire_pos = []
    train_dire_neg = []
    train_rating = []
    train_all_dire_num = []
    train_data_list = []
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        dire_pos = line[9]
        dire_neg = line[10]
        train_user.append(u)
        train_movie.append(i)
        train_rating.append(line[3])
        train_all_dire_num.append(line[6])
        train_dire_pos.append(dire_pos)
        train_dire_neg.append(dire_neg)
        # train_dire_pos.append(0)
        # train_dire_neg.append(0)
    train_data_list.append(train_user)
    train_data_list.append(train_movie)
    train_data_list.append(train_rating)
    train_data_list.append(train_all_dire_num)
    train_data_list.append(train_dire_pos)
    train_data_list.append(train_dire_neg)


    test_user = []
    test_moivie = []
    test_rating = []
    test_dire_all = []
    test_all_dire_num = []
    test_data_list = []
    for line in test_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        dire_pos = line[9]
        dire_neg = line[10]
        test_user.append(u)
        test_moivie.append(i)
        test_rating.append(line[3])
        test_all_dire_num.append(line[6])
        test_dire_all.append(dire_pos + dire_neg)
        # test_dire_pos.append(0)
        # test_dire_neg.append(0)
    test_data_list.append(test_user)
    test_data_list.append(test_moivie)
    test_data_list.append(test_rating)
    test_data_list.append(test_all_dire_num)
    test_data_list.append(test_dire_pos)
    test_data_list.append(test_dire_neg)
    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_data_list, test_data_list, (n_users + 1), (n_items + 1) ,n_dire + 1