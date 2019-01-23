import tqdm
user_dir_score = {}
def outfile(file,list):
    with open(file, 'a+') as f:
        for item in list:
            f.write(item[0] + "\t" + str(item[1]) + "\t" +  str(item[2]) + "\t" +  str(item[3]) + "\n")
def getposneg(scorelist,threshold):
    pos = 0
    neg = 0
    for item in scorelist:
        if item >= threshold:
            pos+=1
        else:
            neg += 1
    return pos,neg

def getstring(scorelist):
    scorestr = ""
    for item in scorelist:
        scorestr += str(item)
    return scorestr

with open("data/ml1m/ratings_date_dire_t_with_dir_num.dat","r") as f:
    for line in tqdm.tqdm(f):
        line_split = line.strip().split("\t")
        try:
            user_dir_score[(int(line_split[0]), int(line_split[6]))].append(int(line_split[2]))
        except:
            user_dir_score[(int(line_split[0]), int(line_split[6]))] = [int(line_split[2])]
with open("data/ml1m/ratings_date_dire_t_with_dir_num.dat","r") as f:
    for line in tqdm.tqdm(f):
        line_split = line.strip().split("\t")
        foutlist = []
        item = []
        item.append(line.strip())
        pos,neg = getposneg(user_dir_score[int(line_split[0]),int(line_split[6])],4)
        item.append(pos)
        item.append(neg)
        item.append(getstring(user_dir_score[int(line_split[0]),int(line_split[6])]))
        foutlist.append(item)
        outfile("data/ml1m/all_1m_ratings.dat",foutlist)