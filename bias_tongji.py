#coding=utf-8
import tqdm
import cPickle
import os
#两个字典
#movie = {movieID,{[looktime(1,2,3..8),scoresum,ratingnum,averag],[looktime(1,2,3..8),scoresum,ratingnum,averag],...,\
# [looktime(1,2,3..8),scoresum,ratingnum,averag]}}
#1	1270	5	978300055	1	1
threshold = 2
movie_rating = {}
if os.path.exists("movie_rating.p"):
    with open("data/ml1m/all_1m_ratings.dat","r") as f:
        for line in tqdm.tqdm(f):
            line_split = line.strip().split("\t")
            if int(line_split[1]) not in movie_rating.keys():
                movie_rating[int(line_split[1])] = {}
                if int(line_split[5]) not in movie_rating[int(line_split[1])].keys():
                    movie_rating[int(line_split[1])][int(line_split[5])] = [int(line_split[2]),1,0]
            else:
                if int(line_split[5]) not in movie_rating[int(line_split[1])].keys():
                    movie_rating[int(line_split[1])][int(line_split[5])] = [int(line_split[2]), 1, 0]
                else:
                    movie_rating[int(line_split[1])][int(line_split[5])][0] += int(line_split[2])
                    movie_rating[int(line_split[1])][int(line_split[5])][1] += 1
cPickle.dump(movie_rating, open("movie_rating.p", "wb"))
movie_rating = cPickle.load(open("movie_rating.p", "rb"))
#开始取平均
for key2,val2 in movie_rating.items():
    for key,val in val2.items():
    #val2[looktime(1,2,3..8),scoresum,ratingnum,averag],[looktime(1,2,3..8),scoresum,ratingnum,averag],...,\
# [looktime(1,2,3..8),scoresum,ratingnum,averag]}
        if val[1] != 0:
            val[2] = (val[0]*1.0)/val[1]
#统计看过人数为2、3、4比1大的比例
proportion = {}
for i in range(1,40):
    proportion[i] = [0,0,0]
    for key2, val2 in movie_rating.items():
        for key, val in val2.items():
            if key == i:
                if 1 in val2.keys():
                    if val2[1][1] != 0 and val[1] != 0:
                        proportion[i][0] += 1
                        if(val[2] > val2[1][2]):
                            proportion[i][1] += 1
for i in range(2,40):
    if proportion[i][0] != 0:
        proportion[i][2] = (proportion[i][1]*1.0)/proportion[i][0]
print proportion
for i in range(2,40):
    print str(i) + "\t" + str(proportion[i][2])
