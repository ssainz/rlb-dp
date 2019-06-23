import config
import os


gamma = 1
N = 100
object_type = 'clk'


data_path = config.ipinyouPath

camp = "1458"

in_file = data_path + camp + "/bid-model/rlb_dnb_gamma={}_N={}_{}.txt".format(gamma,N,object_type)

out_dir = data_path + camp + "/fa-train/rlb_dnb_gamma={}_N={}_{}_1/".format(gamma,N,object_type)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(in_file, "r") as fin:
    count = 0
    for line in fin:
        save_path = out_dir + str(count) + ".txt"
        with open(save_path, "w") as fout:
            fout.write(line+"\n")
        count += 1

