#
#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
import torch

def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P /= sumP

    return H, P


def x2p_torch(X, cuda, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = (-2 * X @ X.t() + sum_X).t() + sum_X

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))

    if cuda:
        P = P.cuda()
        beta = beta.cuda()
        logU = logU.cuda()

    n_list = list(range(n))

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print(f"Computing P-values for point {i} of {n}...")

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]
        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.size()
    X -= torch.mean(X, 0)

    (l, M) = torch.eig(X.t() @ X, True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 1

    Y = X @ M[:, :no_dims]
    return Y


def tsne(X, cuda, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if not isinstance(no_dims, int):
        raise ValueError(f"Error: Expected array X type int, got {type(no_dims)} instead.")

    # Initialize variables
    X = pca_torch(X, initial_dims)
    (n, d) = X.size()
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01

    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    if cuda:
        Y = Y.cuda()
        dY = dY.cuda()
        iY = iY.cuda()
        gains = gains.cuda()

    # Compute P-values
    P = x2p_torch(X, cuda, 1e-5, perplexity)
    P += P.t()
    P /= torch.sum(P)
    P *= 4.    # early exaggeration
    print(f"P is of shape {P.shape}")
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * Y @ Y.t()
        num = 1. / (1. + (num + sum_Y).t() + sum_Y)
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        momentum = initial_momentum if iter < 20 else final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y += iY
        Y -= torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print(f"Iteration {iter+1}: error = {C}")

        # Stop lying about P-values
        if iter == 100:
            P /= 4.

    # Return solution
    return Y