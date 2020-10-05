
import os
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch

from torch_tsne import tsne

mnist = osp.join(osp.abspath(osp.dirname(__file__)), "..", "mnist")

parser = argparse.ArgumentParser()
parser.add_argument("--xfile", type=str, default=osp.join(mnist, "mnist2500_X.txt"), help="file name of feature stored")
parser.add_argument("--yfile", type=str, default=osp.join(mnist, "mnist2500_labels.txt"), help="file name of label stored")
parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")

opt = parser.parse_args()
print("get choice from args", opt)
xfile = opt.xfile
yfile = opt.yfile

if opt.cuda:
    print("set use cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    X = np.loadtxt(xfile)
    X = torch.Tensor(X)
    labels = np.loadtxt(yfile).tolist()

    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:,0])==len(X[:,1]))
    assert(len(X)==len(labels))

    with torch.no_grad():
        Y = tsne(X, 2, 50, 20.0)

    if opt.cuda:
        Y = Y.cpu().numpy()

    # You may write result in two files
    # print("Save Y values in file")
    # Y1 = open("y1.txt", 'w')
    # Y2 = open('y2.txt', 'w')
    # for i in range(Y.shape[0]):
    #     Y1.write(str(Y[i,0])+"\n")
    #     Y2.write(str(Y[i,1])+"\n")

    torch.save(Y, osp.join("results", "tsne_coords.pt"))

    pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pyplot.show()
