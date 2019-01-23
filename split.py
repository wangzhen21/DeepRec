import random
with open("data/ml1m_temp/train_1m_ratings_temp.dat","wb") as f_train:
    with open("data/ml1m_temp/test_1m_ratings_temp.dat","wb") as f_test:
        with open("data/ml1m/ratings_date_dire_t_with_dir_num.dat","rb") as f_in:
            for line in f_in:
                i = random.randint(0,9)
                if i == 0:
                    f_test.write(line)
                else:
                    f_train.write(line)