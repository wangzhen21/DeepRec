import random
import tqdm
import os

def outfile(file,outstring):
    with open(file, 'a+') as f:
            f.write(outstring + "\n")
def delfile(file):
    if os.path.exists(file):
        os.remove(file)

delfile("data/ml1m/nfm_valid_1m_ratings.dat")
delfile("data/ml1m/nfm_test_1m_ratings.dat")
with open("data/ml1m/test_1m_ratings.dat","r") as fin:
    for line in tqdm.tqdm(fin):
        line = line.strip()
        i = random.randint(1,2)
        if i == 1:
            outfile("data/ml1m/nfm_valid_1m_ratings.dat", line)
        else:
            outfile("data/ml1m/nfm_test_1m_ratings.dat", line)
            #pass