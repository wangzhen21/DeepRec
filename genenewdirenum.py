#coding=utf-8
import tqdm
import os
features = {}
iiii = 0
def delfile(file):
    if os.path.exists(file):
        os.remove(file)

def outfile(file,outstring):
    with open(file, 'a+') as f:
            for line in f:
                line_split = line.strip().split('\t')
            f.write(outstring + "\n")

def fin_fout_file(filein,fileout):
    '''
    let the input file to outputfile
    :param filein:
    :param fileout:
    :return:
    '''
    with open(filein,'r') as fin:
        for line in tqdm.tqdm(fin):
            line_split = line.strip().split('\t')
            outfile(fileout,line_split[0] + '\t' + line_split[1] + '\t' + line_split[2] + '\t' + str(features[(line_split[3],line_split[4])]) + ":3")

def getfeatureindex(file,iiii):
    with open(file,'r') as fin:
        for line in tqdm.tqdm(fin):
            line_split = line.strip().split('\t')
            if (line_split[3],line_split[4]) not in features.keys():
                features[(line_split[3],line_split[4])] = iiii
                iiii = iiii + 1

getfeatureindex("data/ml1m/train_1m_ratings.dat.nfm",iiii)
getfeatureindex("data/ml1m/test_1m_ratings.dat.nfm",iiii)
delfile("data/ml1m_nfm/nfm_train_1m_ratings.dat")
delfile("data/ml1m_nfm/nfm_test_1m_ratings.dat")

fin_fout_file("data/ml1m/train_1m_ratings.dat.nfm","data/ml1m_nfm/nfm_train_1m_ratings.dat")
fin_fout_file("data/ml1m/test_1m_ratings.dat.nfm","data/ml1m_nfm/nfm_test_1m_ratings.dat")