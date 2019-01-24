#coding=utf-8
import tqdm
import cPickle
import os
#两个字典
#movie = {movieID,{[looktime(1,2,3..8),scoresum,ratingnum,averag],[looktime(1,2,3..8),scoresum,ratingnum,averag],...,\
# [looktime(1,2,3..8),scoresum,ratingnum,averag]}}
#1	1270	5	978300055	1	1

#分别统计统计电影平均分，负用户 和 正用户的评分与 一部都没看过的平均分比较。r- r r+
#则需要三个字典 分别记录 每个电影 对应的 三类用户 评分的平均分
#每一个字典的结构为
# movie_num:[score_sum,rating_sum,average]
#一个字典用于记录没看过此导演其他电影的用
#一个用于记录打分全是正向的评分 或者 (1、2、3)分比例 小于 threshold 的,属于正用户
# 其他用户 属于 负用户一个用于记录曾经打过(1、2、3)分数量 超过n(1 或者其他)的平均分
#1	1270	5	978300055	1	1	709	Robert Zemeckis	0	0	5
def add_to_movie_rating(line_split,movie_rating):
    if int(line_split[1]) not in movie_rating.keys():
        movie_rating[int(line_split[1])] = [0,0,0]
    movie_rating[int(line_split[1])][0] += int(line_split[2])
    movie_rating[int(line_split[1])][1] += 1
movie_rating_single = {}
movie_rating_neg = {}
movie_rating_pos = {}
threshold = 0.1
movie_rating = {}
if os.path.exists("movie_rating.p"):
    with open("data/ml1m/all_1m_ratings_minus.dat","r") as f:
        for line in tqdm.tqdm(f):
            line_split = line.strip().split("\t")
            #首先, 获取此用户属于哪一类
            if (int(line_split[8]) + int(line_split[9])) == 0:
                add_to_movie_rating(line_split, movie_rating_single)
                continue
            # if int(line_split[8]) == 0:
            #     add_to_movie_rating(line_split, movie_rating_neg)
            #     continue
            # if  float(line_split[9])/float(line_split[8]) < 0.1:
            #     add_to_movie_rating(line_split, movie_rating_pos)
            #     continue
            if int(line_split[9]) <= int(line_split[8]):
                add_to_movie_rating(line_split, movie_rating_pos)
                continue
            add_to_movie_rating(line_split, movie_rating_neg)
movie_rating_list = [movie_rating_single,movie_rating_neg,movie_rating_pos]
for item in movie_rating_list:
    for key in item.keys():
        item[key][2] = float(item[key][0])/float(item[key][1])
#用来存储一个比例
#总数/大于平均分的个数/小于平均分的个数
proportion_pos = [0,0,0]
proportion_neg = [0,0,0]
for key in movie_rating_single.keys():
    if key in movie_rating_neg.keys():
        proportion_neg[0] += 1
        if movie_rating_neg[key][2] > movie_rating_single[key][2]:
            proportion_neg[1] += 1
    if key in movie_rating_pos.keys():
        proportion_pos[0] += 1
        if movie_rating_pos[key][2] > movie_rating_single[key][2]:
            proportion_pos[1] += 1
proportion_pos[2] = proportion_pos[0] - proportion_pos[1]
proportion_neg[2] = proportion_neg[0] - proportion_neg[1]
print "proportion_pos\n"
print proportion_pos
print "proportion_neg\n"
print proportion_neg