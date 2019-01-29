import random
import tqdm
import os

def outfile(file,outstring):
    with open(file, 'a+') as f:
            f.write(outstring + "\n")
def delfile(file):
    if os.path.exists(file):
        os.remove(file)

def cutfile(file):
    delfile(file + ".cut_dire")
    with open(file,"r") as fin:
        for line in tqdm.tqdm(fin):
            line = line.strip()
            i = random.randint(1,1)
            if i == 1:
                line_split = line.split('\t')
                outstring = line_split[0] + '\t' + line_split[1] + '\t' + line_split[2]
                outfile(file + ".cut_dire", outstring)
            else:
                #outfile("data/ml1m/nfm_test_1m_ratings.dat", line)
                pass
cutfile("data/ml1m/nfm_test_1m_ratings.dat")
cutfile("data/ml1m/nfm_train_1m_ratings.dat")
cutfile("data/ml1m/nfm_valid_1m_ratings.dat")