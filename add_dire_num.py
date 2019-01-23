import cPickle
import os
import random
import tqdm
if os.path.exists("data/ml1m/ratings_date_dire_t_with_dir_num.dat"):
    os.remove("data/ml1m/ratings_date_dire_t_with_dir_num.dat")
movielen_dire_full_time = cPickle.load(open("data/ml1m/movielid_dir_name_full_movie.p", "rb"))
num_all = 0
with open("data/ml1m/ratings_date_dire_t.dat","rb") as f:
    with open("data/ml1m/ratings_date_dire_t_with_dir_num.dat","wb") as fout:
        for line in tqdm.tqdm(f):
            line = line.strip()
            line_split = line.strip().split("\t")
            if int(line_split[1]) in movielen_dire_full_time.keys():
                line = line + "\t" + str(movielen_dire_full_time[int(line_split[1])][0]) + '\t' + movielen_dire_full_time[int(line_split[1])][1]
            else:
                num_all += 1
                line = line + "\t" + str(random.randint(1,10000)) + "\t" + "None"
            fout.write(line + "\n")
print num_all