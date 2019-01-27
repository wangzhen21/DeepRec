import os
import tqdm
def outfile(file,outstring):
    with open(file, 'a+') as f:
            f.write(outstring + "\n")
#1	2028	5	978301619	1	3	229	Steven Spielberg	2	0	545
def gene(fullfile):
    if os.path.exists(fullfile + ".nfm"):
        os.remove(fullfile + ".nfm")
    with open(fullfile,'r') as fin:
        for line in tqdm.tqdm(fin):
            line_split = line.strip().split('\t')
            out_string = line_split[2] + "\t" + line_split[0] + ":1" + "\t" + line_split[1] + ":2" + "\t" + \
                         line_split[8] + ":3" + "\t" + line_split[9] + ":4"
            outfile(fullfile + ".nfm", out_string)
gene("data/ml1m/train_1m_ratings.dat")
gene("data/ml1m/test_1m_ratings.dat")