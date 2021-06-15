import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
from colors import bcolors
import glob
import os

def plot_2d_feat(args):
    # Parse args
    checkpoint_dir = args.checkpoint_dir
    d1 = args.dim_1
    d2 = args.dim_2
    
    for npz_file_path in glob.glob(checkpoint_dir+"*task0.npz"):
        base_name = os.path.basename(npz_file_path).split(".")[0]
        # data loading
        train_data = np.load(npz_file_path)
        # image & label
        train_label_data = train_data['label']
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkred', 'orange', 'darkblue']
        # load file
        data = train_data["z_query"]
        n_query = data.shape[0]

        plt.figure(figsize=(5, 5))
        for i in range (n_query):
            a = int(train_label_data[i])
            z_d1 = data[i][d1]
            z_d2 = data[i][d2]
            plt.scatter(z_d1, z_d2, s=30, color=colors[a])
        # for x in range(0, 10):
        #     plt.scatter(0, 0, color=colors[x], label=str(x))
        plt.title("Latent Features for "+base_name)
        # plt.legend()
        plt.savefig(checkpoint_dir+"/"+base_name+'_features.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("-", "--", default = , help="")
    parser.add_argument("-p", "--checkpoint_dir", help="Path to checkpoint_dir")
    parser.add_argument("-d1", "--dim_1", required=False, default=0, type=int, help="Index of dimension to be selected as the first dim. to plot")
    parser.add_argument("-d2", "--dim_2", required=False, default=1, type=int, help="Index of dimension to be selected as the second dim. to plot")
    args = parser.parse_args()
    if args.dim_1 >= 512 or args.dim_2 >= 512 or args.dim_1 < 0 or args.dim_2 < 0:
        print(bcolors.FAIL + "[Error]" + bcolors.ENDC + " Out of available dims. Valid dim number: 0-511")
        exit(1)
    if args.dim_1 == args.dim_2:
        print(bcolors.WARNING + "[WARNING]" + bcolors.ENDC + " Same dimension selected")
    plot_2d_feat(args)