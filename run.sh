#!/bin/bash
trap "exit" INT
dataset=${1?:Error: mnist/sanitizer}
N=${2?:Error: the number of components}
mu_prior=${3?:Error: the priors, single value 0/1}
infinity=${4:-false}
T_gibb=10000

version=0
if [ "$infinity" = false ]; then
  sigma_prior_method=limited
  for sigma_prior in 10 100 1000 10000 100000
    do
    python3 run_nmf_with_diff_prior.py --dataset "$dataset" --N "$N" --T_gibb "$T_gibb" \
                                     --sigma_a_prior "$sigma_prior_method" --sigma_a_prior_value "$sigma_prior" \
                                     --sigma_b_prior "$sigma_prior_method" --sigma_b_prior_value "$sigma_prior" \
                                     --mu_a_prior "$mu_prior" --mu_b_prior "$mu_prior" --version "$version"

  done
fi

if [ "$infinity" = true ]; then
  sigma_prior_method=infinity
  python3 run_nmf_with_diff_prior.py --dataset "$dataset" --N "$N" --T_gibb "$T_gibb" \
                                     --sigma_a_prior "$sigma_prior_method" --sigma_a_prior_value 0 \
                                     --sigma_b_prior "$sigma_prior_method" --sigma_b_prior_value 0 \
                                     --mu_a_prior "$mu_prior" --mu_b_prior "$mu_prior" --version "$version"
fi