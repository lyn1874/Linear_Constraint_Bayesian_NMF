"""
Created on 10:24 at 14/12/2021
@author: bo 
"""
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_args():
    parser = argparse.ArgumentParser(description='NMF')
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument("--T_gibb", type=int, default=5000)
    parser.add_argument("--N", type=int, default=40, help="The number of components")
    parser.add_argument("--sigma_a_prior", type=str, default="infinity")
    parser.add_argument("--sigma_b_prior", type=str, default="infinity")
    parser.add_argument("--sigma_a_prior_value", type=int, default=1)
    parser.add_argument("--sigma_b_prior_value", type=int, default=1)
    parser.add_argument("--mu_a_prior", type=float, default=0)
    parser.add_argument("--mu_b_prior", type=float, default=0)
    parser.add_argument("--version", type=int, default=0)
    return parser.parse_args()

