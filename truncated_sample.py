"""
Created on 18:10 at 07/12/2021
@author: bo 
"""
import numpy as np
import scipy.io as io


def randstgs(l, u, x, test=False):
    """Generate random numbers from truncated Gaussian density
    p(x) = const * np.exp(-x**2/2)I(l<x<u)
    Args:
        l: [1, num_of_samples]
        u: [1, num_of_samples]
        x: [1, num_of_samples]
        test: bool variable, set True when I want to check my implementation and Mikkel's matlab implementation
    """
    value = [np.shape(v) for v in [l, u, x]]
    ndims = [len(v) for v in value]
    group_index = np.concatenate([np.arange(v) for v in ndims])
    unique_group_index = np.unique(group_index)
    value = np.concatenate(value, axis=0)
    sz = []
    for i in unique_group_index:
        select_index = np.where(group_index == i)[0]
        select_value = np.max(value[select_index])
        sz.append(select_value)
    sz = np.array(sz)
    _mean = -0.5 * x ** 2
    if test:
        mm1 = np.ones(np.prod(sz)).reshape(sz)
        mm2 = np.ones(np.prod(sz)).reshape(sz)
    else:
        mm1 = np.random.rand(np.prod(sz)).reshape(sz)
        mm2 = np.random.rand(np.prod(sz)).reshape(sz)
    _std = np.log(mm1)  #
    z = _mean + _std
    s = np.sqrt(-2 * z)
    ll = np.array([np.max([v, q]) for v, q in zip(-s, l)])
    uu = np.array([np.min([v, q]) for v, q in zip(s, u)])
    x = mm2 * (uu - ll) + ll # np.random.rand(np.prod(sz)).reshape(sz)
    return x


def randcg(mx, Sxx, A, b, Aeq, beq, x0, T, test):
    """Generate random numbers from the constrained Gaussian distribution
    p(x) = const * N(mx, Sxx), s.t. A * x - b >0, Aeq * x - beq = 0
    Args:
        mx: mean vector
        Sxx: convariance matrix
        A, b: matrix and vector that specify the inequality constraint
        Aeq, beq: matrix and vector that specify the equality constraint
    """

    # 1. get the dimensionality of the data
    N = np.shape(mx)[0]
    eps = 2.2204e-16
    # 2. get the orthogonal basis of the equality constraints and the new origin
    if len(Aeq) == 0:
        P = np.zeros([0, N])
        K = np.eye(N)
        cx = np.zeros([N, 1])
    else:
        [U, S, V] = np.linalg.svd(Aeq.T)  # Eq.16
        m, n = np.shape(Aeq)
        if m > 1:
            s = np.diag(S)
        elif m == 1:
            s = S[0]
        else:
            s = 0
        tot = np.max([m, n]) * np.max(s) * eps
        r = np.sum(s > tot)
        P = U[:, 0:r].T
        K = U[:, r:].T
        cx = np.zeros([len(mx), 1])
        cx[0] = 1   # this is equivalent to \ operation in matlab to solve Ax=b

    # 3. Deal with the inequality conditions
    if len(A) == 0:
        A = np.zeros([0, N])
        b = np.zeros([0, 1])

    M = N - np.shape(P)[0]  # dimension of the space that satisfy the equality Eq. 17
    # 4. Get the first equation in page 6
    if len(Aeq) > 0:
        _first_part = np.matmul(Sxx, P.T)
        _second_part = np.matmul(np.matmul(P, Sxx), P.T)
        _third_part = 1 / _second_part * P
        _tot = np.matmul(_first_part, _third_part)
    else:
        _tot = np.zeros([N, N])
    W = np.matmul(K, (np.eye(N) - _tot))
    my = np.matmul(W, (mx - cx))
    linside = np.matmul(np.matmul(W, Sxx), K.T) # This correspond to SigamY Eq.19.2
    L = np.linalg.cholesky(linside)  # This is needed for Eq.22
    L = L.T
    # Starting point
    starting_point_eq17 = np.matmul(K, x0 - cx)
    starting_point_eq20 = starting_point_eq17 - my
    w = np.matmul(np.linalg.inv(L.T), starting_point_eq20) # This correspond to equation 20

    # Precomputations for bounds Eq 22
    E = np.matmul(np.matmul(A, K.T), L.T)
    e = b - np.matmul(A, np.matmul(K.T, my) + cx)  # what is this?

    for t in range(T):
        for m in range(M):
            nm = np.delete(np.arange(M), m)

            # compute the lower bound and upper bound again
            n = e - np.matmul(E[:, nm], w[nm, :])
            d = E[:, m:m+1]
            d_g0_index = np.where(d>0)[0]
            d_s0_index = np.where(d<0)[0]
            lb_divide = n[d_g0_index, :] / d[d_g0_index, :]
            ub_divide = n[d_s0_index, :] / d[d_s0_index, :]
            lb = np.max(lb_divide, axis=0)
            if len(lb) == 0:
                lb = -np.inf
            ub = np.min(ub_divide, axis=0)
            if len(ub) == 0:
                ub = np.inf
            w[m, :] = randstgs(lb, ub, w[m, :], test)
    x = np.matmul(K.T, np.matmul(L.T, w) + my) + cx
    return x


def validate_with_matlab_implementation():
    mx = io.loadmat("data_for_check_implementation/mx.mat")["mx"]
    x0 = io.loadmat("data_for_check_implementation/x0.mat")["x0"]
    Sxx = io.loadmat("data_for_check_implementation/Sxx.mat")["Sxx"]
    final_output = io.loadmat("data_for_check_implementation/output_randcg.mat")["x"]
    A = np.eye(3)
    b = np.zeros([3, 1])
    Aeq = np.ones([1, 3])
    beq = np.ones([1, 1])
    x_my_out = randcg(mx, Sxx, A, b, Aeq, beq, x0, T=1, test=True)
    print("The shape of my output", x_my_out.shape)
    print("The shape of matlab implementation", final_output.shape)
    print("The difference", np.sum(x_my_out - final_output))






