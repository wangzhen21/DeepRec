from tqdm import tqdm
def outfile(file,outstring):
    with open(file, 'a+') as f:
            f.write(outstring + "\n")

with open("data/ml1m/train_1m_ratings.dat","r") as f_in:
    for line in tqdm(f_in):
        line_split = line.strip().split('\t')
#1	1270	5	978300055	1	1	709	Robert Zemeckis	1	0	5
        if int(line_split[2]) >= 4:
            outstring = line_split[0] + "\t" + line_split[1]+ "\t" + line_split[2] +  "\t" + line_split[3] +\
                        "\t" + line_split[4] + "\t" + line_split[5] + "\t" + line_split[6] + "\t" + line_split[7] + \
                        "\t" + str(int(line_split[8]) - 1) + "\t" + line_split[9] + "\t" + line_split[10]
            outfile("data/ml1m/train_1m_ratingstemp.dat", outstring)
        else:
            outstring = line_split[0] + "\t" + line_split[1] + "\t" + line_split[2] + "\t" + line_split[3] + \
                        "\t" + line_split[4] + "\t" + line_split[5] + "\t" + line_split[6] + "\t" + line_split[7] + \
                        "\t" + line_split[8] + "\t" + str(int(line_split[9]) - 1) + "\t" + line_split[10]
            outfile("data/ml1m/train_1m_ratingstemp.dat", outstring)