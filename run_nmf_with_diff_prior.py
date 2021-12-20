import numpy as np
import os
import scipy.io as io
import truncated_sample as ts
# from tqdm import tqdm
import pickle
import sanitizer as sanitizer
import config as config


def nmf(args, data, tds_dir):
    """Run the NMF algorithm
    Args:
        args: arguments
        data: [I, J] I -> number of features J -> number of samples
        tds_dir: the directory to save the data
    """
    N = args.N
    T_gibb = args.T_gibb
    sigma_a_prior_value = args.sigma_a_prior_value
    sigma_b_prior_value = args.sigma_b_prior_value

    sigma_a_prior = args.sigma_a_prior
    sigma_b_prior = args.sigma_b_prior

    mu_a_prior = args.mu_a_prior
    mu_b_prior = args.mu_b_prior

    I, J = np.shape(data)

    # create A inequality condition
    AA = np.concatenate([np.eye(N), -np.eye(N)], axis=0)  # the inequality condition for A matrix
    bA = np.concatenate([np.zeros([N, 1]), -1 * np.ones([N, 1])], axis=0)
    # operation: it contain two identity matrix. The first identity matrix helps to make sure that A is non-negative,
    # the second identity matrix makes sure that A is between 0 and 1. Because if a single value in A is 10, -10 < -1,
    # the second condition will not work.

    # create B inequality condition
    BA = np.eye(N)  # We need to make sure each coefficient is larger than 0
    Bb = np.zeros([N, 1])
    # operation: if N = 2, then BA * B[:, 0] = [[1, 0],[0, 1]] * [2, 3] = [2, 3] > [0, 0]

    # create B equality conditions
    BAeq = np.ones([1, N])  # The coefficient of each coefficient should also be 1
    Bbeq = np.ones([1, 1])  # The sum of the b coefficient per sample = 1
    # operation: BAeq * B[:, 0] = Bbeq -> This is the equality condition is met

    # initialisation
    x = np.sum(data ** 2) / 2

    A = np.random.rand(I, N)  # [number_of_feature, number_of_components]
    B = np.random.rand(N, J)  # [number_of_components, number_of_samples]
    B = B / np.sum(B, axis=0, keepdims=True)  # [number_of_components, number_of_samples]

    sigma = 1
    sigma_A = np.eye(N) * sigma_a_prior_value
    sigma_B = np.eye(N) * sigma_b_prior_value
    mu_a = np.ones_like(A) * mu_a_prior
    mu_b = np.ones_like(B) * mu_b_prior

    if sigma_a_prior_value != 0 and sigma_b_prior_value != 0:
        print("---------------------------")
        print("Sigma A value", np.unique(sigma_A), np.unique(np.linalg.inv(sigma_A)))
        print("Sigma B value", np.unique(sigma_B), np.unique(np.linalg.inv(sigma_B)))
        print("Mu a value", np.unique(mu_a))
        print("Mu b value", np.unique(mu_b))
        print("---------------------------")

    str_use = ["A", "B", "Error"]
    MSE_g = []
    data_statisitics = {}
    data_statisitics["A_mu"] = []
    data_statisitics["A_sigma"] = []
    data_statisitics["B_mu"] = []
    data_statisitics["B_sigma"] = []
    data_statisitics["sigma"] = []
    for t in range(T_gibb):
        # Generate A
        if sigma_a_prior == "infinity":
            C = np.matmul(B, B.T)
        else:
            C = np.matmul(np.linalg.inv(sigma_A), np.eye(N) * sigma) + np.matmul(B, B.T)

        if mu_a_prior == 0 or sigma_a_prior == "infinity":
            D = np.matmul(data, B.T)
        else:
            D = np.matmul(mu_a, np.linalg.inv(sigma_A)) * sigma + np.matmul(data, B.T)

        C_inv = np.linalg.inv(C)
        Mu = np.matmul(D, C_inv)
        S = np.matmul(sigma * np.eye(N), C_inv)

        A = ts.randcg(Mu.T, S.T, AA, bA, [], [], A.T, 1, False).T
        if t % 10 == 0:
            data_statisitics["A_mu"].append(Mu)
            data_statisitics["A_sigma"].append(S)
        # Generate noise variance
        sigma_scale = I * J / 2 + 1
        sigma_shape = 1 / (x + np.sum(np.sum(np.multiply(A, (np.matmul(A, C) - 2 * D)))) / 2)
        sigma = 1 / np.random.gamma(sigma_scale, sigma_shape)
        # sigma = 0.12
        if t % 10 == 0:
            data_statisitics["sigma"].append(sigma)
        # Generate B
        if sigma_b_prior == "infinity":
            E = np.matmul(A.T, A)
        else:
            E = np.matmul(np.linalg.inv(sigma_B), np.eye(N) * sigma) + np.matmul(A.T, A)

        if mu_b_prior == 0 or sigma_b_prior == "infinity":
            F = np.matmul(A.T, data)
        else:
            F = np.matmul(np.linalg.inv(sigma_B), mu_b) * sigma + np.matmul(A.T, data)

        E_inv = np.linalg.inv(E)
        Mu = np.matmul(E_inv, F)

        S = np.matmul(E_inv, sigma * np.eye(N))

        B = ts.randcg(Mu, S, BA, Bb, BAeq, Bbeq, B, 1, False)
        if t % 10 == 0:
            data_statisitics["B_mu"].append(Mu)
            data_statisitics["B_sigma"].append(S)
        _mse_error = np.sum((data - np.matmul(A, B))**2)
        MSE_g.append(_mse_error)
        if t % 10 == 0:
            for s_str, s_value in zip(str_use, [A, B, MSE_g]):
                np.save(tds_dir + "/%s" % s_str, np.array(s_value))
            with open(tds_dir + "/mean_and_covariance_stat.obj", "wb") as f:
                pickle.dump(data_statisitics, f)
    return A, B, MSE_g


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    args = config.give_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    model_mom = "../exp_data/%s/" % args.dataset
    model_dir = model_mom + "/version_%d_sigma_prior_A_%s_%d_B_%s_%d" % (args.version,
                                                                         args.sigma_a_prior, args.sigma_a_prior_value,
                                                                         args.sigma_b_prior, args.sigma_b_prior_value)
    model_dir += "_mu_a_prior_%d_mu_b_prior_%d" % (args.mu_a_prior, args.mu_b_prior)
    model_dir += "_Tsteps_%d_num_component_%d/" % (args.T_gibb, args.N)
    create_dir(model_dir)
    if args.dataset == "mnist":
        data = io.loadmat("mixeddigits.mat")["X"]
    elif args.dataset == "sanitizer":
        data, _, _ = sanitizer.prepare_mixture_data(model_dir)
    nmf(args, data, model_dir)















