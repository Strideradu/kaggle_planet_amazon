import random

file_name = []
with open("D:/Data/kaggle_amazon/sample_submission_v2.csv") as f:
    for line in f.readlines():
        if line[0]!="i":
            file_name.append(line.split(",")[0])
"""
valid_3000 = random.sample(file_name, 3000)
train_37479 = list(set(file_name).difference(valid_3000))
train_40479 = file_name
"""
def writeline_to_file(list, file_path):
    with open(file_path, "w") as fout:
        for i in list:
            fout.write(i)
            fout.write("\n")
writeline_to_file(file_name, "D:/Data/kaggle_amazon/test-61191")
#writeline_to_file(valid_3000, "D:/Data/kaggle_amazon/valid-3000")
#writeline_to_file(train_37479, "D:/Data/kaggle_amazon/train-37479")
#writeline_to_file(train_40479, "D:/Data/kaggle_amazon/train-40479")