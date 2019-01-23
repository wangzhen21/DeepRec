with open("data/ml1m/ratings.dat") as f:
    with open("data/ml1m/ratings_t.dat","w") as fout:
        for line in f:
            line = line.strip().replace("::","\t")
            fout.write(line + "\n")