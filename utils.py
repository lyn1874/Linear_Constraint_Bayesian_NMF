"""
Created on 09:35 at 15/12/2021
@author: bo 
"""
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import sanitizer as load_sanitizer
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


def get_model_dir(gpu_machine, dataset, sigma_prior, sigma_prior_value, mu_prior_value, Tsteps, num_component,
                  version=0):
    sigma_a_prior, sigma_b_prior = sigma_prior
    sigma_a_prior_value, sigma_b_prior_value = sigma_prior_value
    mu_a_prior, mu_b_prior = mu_prior_value
    if gpu_machine:
        tds_mom = "../exp_data/BDA_experiment/%s/" % dataset
    else:
        tds_mom = "../exp_data/%s/" % dataset
    files = [v for v in os.listdir(tds_mom) if "num_component" in v]
    files = [v for v in files if "sigma_prior_A_%s_%d_B_%s_%d" % (sigma_a_prior, sigma_a_prior_value,
                                                                  sigma_b_prior, sigma_b_prior_value) in v]
    files = [v for v in files if "mu_a_prior_%d_mu_b_prior_%d" % (mu_a_prior, mu_b_prior) in v]
    files = [v for v in files if "Tsteps_%d_num_component_%d" % (Tsteps, num_component) in v]
    files = [v for v in files if "version_%d" % version in v]
    if len(files) == 1:
        return tds_mom + files[0] + "/"
    else:
        return [tds_mom + v + "/" for v in files]


def compare_mse_loss_over_diff_component(sigma_prior_value, mu_value, dataset, component_g):
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    color_group = ['r', 'g', 'b', 'c', "m", "y", "orange"]
    for i, s_comp in enumerate(component_g):
        if sigma_prior_value == 0:
            method = ["infinity", "infinity"]
        else:
            method = ["limited", "limited"]
        tds_dir = get_model_dir(False, dataset, method, [sigma_prior_value, sigma_prior_value],
                                [mu_value, mu_value], 10000, s_comp, 0)
        error = np.load(tds_dir + "Error.npy")
        if dataset == "mnist":
            error = error / 4000
        x_value = np.arange(len(error))[1000:]
        label = "N = %d" % s_comp
        ax.plot(x_value, error[1000:], color=color_group[i], label=label)
    ax.legend(loc='best')
    ax.set_xlabel("Gibbs steps", fontsize=10)
    if "sanitizer" in tds_dir:
        ax.set_ylabel("MSE per spectrum", fontsize=10)
    else:
        ax.set_ylabel("MSE per image", fontsize=10)


def compare_mse_loss(sigma_prior_value, mu_value, dataset, dir2save, save=False):
    fig = plt.figure(figsize=(8, 3))
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    title_use = ["N=4", "N=8"]
    color_group = ['r', 'g', 'b', 'c', "m", "y", "orange"]
    line_group = ['-', ':']

    for k_index, k in enumerate([4, 8]):
        ax = fig.add_subplot(1, 2, k_index + 1)
        for i, s_sigma in enumerate(sigma_prior_value):
            if s_sigma == 0:
                mu_use = [0]
            else:
                mu_use = mu_value
            for j, s_mu in enumerate(mu_use):
                if s_sigma == 0:
                    method = ["infinity", "infinity"]
                else:
                    method = ["limited", "limited"]
                tds_dir = get_model_dir(False, dataset, method, [s_sigma, s_sigma], [s_mu, s_mu],
                                        10000, k, 0)
                error = np.load(tds_dir + "Error.npy")
                if dataset == "mnist":
                    error = error / 4000
                x_value = np.arange(len(error))[1000:]
                label = r'$\sigma$' + " = %.1E" % s_sigma + " " + r'$\mu$' + " =%d " % (s_mu)
                if s_sigma == 0:
                    label = r'$\sigma$' + " = " + r'$\infty$' + " " + r'$\mu$' + " =%d " % (s_mu)
                ax.plot(x_value, error[1000:], color=color_group[i], ls=line_group[j], label=label)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title(title_use[k_index], fontsize=10)
        if k_index == 1:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    fig.tight_layout()
    ax_global.set_xlabel("\n\nGibbs steps", fontsize=10)
    if "sanitizer" in tds_dir:
        ax_global.set_ylabel("MSE per spectrum\n\n", fontsize=10)
    else:
        ax_global.set_ylabel("MSE per image\n\n", fontsize=10)
    plt.subplots_adjust(wspace=0.2)
    if save:
        plt.savefig(dir2save + "/error_convergence.pdf", pad_inches=0, bbox_inches='tight')


def get_label_for_each_component(A, o_spectra, cls_label):
    if np.max(o_spectra) != 0:
        o_spectra = o_spectra / np.max(o_spectra, axis=1, keepdims=True)
    avg = np.array(o_spectra[:36])
    cos_sim = cosine_similarity(avg, A.T).T
    label = cls_label[:36][np.argsort(cos_sim, axis=-1)[:, -9:]]  # each alcohol compound has 9 spectra
    label_update = []
    for i_index, k in enumerate(label):
        k_unique, k_freq = np.unique(k, return_counts=True)
        if len(k_unique) != 1:
            label_update.append(k_unique[np.argsort(k_freq)[-2:]])
        else:
            label_update.append(k_unique[0])
    label_update = np.array(label_update)
    return cos_sim, label_update


def get_nmf_representation(sigma_prior_value, mu_value, num_component, dataset, save=False):
    if sigma_prior_value == 0:
        method = ["infinity", "infinity"]
    else:
        method = ["limited", "limited"]
    tds_dir = get_model_dir(False, dataset, method, [sigma_prior_value, sigma_prior_value],
                            [mu_value, mu_value],
                            10000, num_component, 0)
    nmf_comp = np.load(tds_dir + "/A.npy")
    alcohol = pickle.load(open(tds_dir + "/alcohol_data.obj", "rb"))
    o_spectra = alcohol["one_component_spectra"]
    cls_name = np.array(['2-propanol', 'Ethanol', 'Methanol', '1-propanol'])
    o_spectra_norm = o_spectra / np.max(o_spectra, axis=1, keepdims=True)
    mix_spectra = alcohol["mixture_data"].T
    wavenumber = alcohol["wavenumber"]
    # model = NMF(n_components=4, init='random', random_state=0).fit(alcohol["mixture_data"].T)
    # ica_comp = model.components_.T
    color_use = ['r', 'g', 'b', 'm']
    one_comp_label = np.concatenate([np.zeros([9]) + i for i in range(4)], axis=0)
    fig = plt.figure(figsize=(9, 1.8))
    for i, comp in enumerate([nmf_comp]):
        _cos_sim, _ = get_label_for_each_component(comp, o_spectra_norm, one_comp_label)
        ax = fig.add_subplot(1, 2, i*2 + 1)
        ax.imshow(enlarge_visualize(_cos_sim, 2))
        ax.set_xticks(np.linspace(0, 27, 4) + 4.5)
        ax.set_xticklabels(cls_name)
        ax.set_yticks(np.linspace(0, len(_cos_sim) * 2 - 1, num_component))
        ax.set_yticklabels(["C%d" % i for i in range(num_component)])
        ax.set_xlabel("Reference spectra", fontsize=10)
        ax.set_ylabel("Extracted \n component", fontsize=10)
        ax = fig.add_subplot(1, 2, i * 2 + 2)
        for j, s_comp in enumerate(comp.T):
            ax.plot(wavenumber, s_comp, color=color_use[j], label="C%d" % j)
        ax.legend(loc='best')
        ax.set_xlabel("Wavenumber(cm" + r'$^{-1}$' + ")", fontsize=10)
        ax.set_ylabel("A.U.", fontsize=10)
    plt.subplots_adjust(wspace=0.2)
    fig.tight_layout()
    if save:
        plt.savefig(tds_dir + "/cos_similarity_comp_tg_component.pdf", pad_inches=0, bbox_inches='tight')


def calc_cosine_similarity(tds_dir, ax, cls_name):
    A = np.load(tds_dir + "A.npy")
    alcohol = pickle.load(open(tds_dir + "/alcohol_data.obj", 'rb'))
    o_spectra_norm = alcohol["one_component_spectra"] / np.max(alcohol["one_component_spectra"], axis=1,
                                                               keepdims=True)
    avg = np.array(o_spectra_norm[:36])
    cos_sim = cosine_similarity(avg, A.T)
    if len(A.T) == 4:
        sim_matrix = enlarge_visualize(cos_sim.T, 2)
    else:
        sim_matrix = cos_sim.T
    ax.imshow(sim_matrix)
    ax.set_xticks(np.linspace(0, 27, 4) + 4.5)
    ax.set_xticklabels(cls_name)
    ax.set_yticks(np.linspace(0, len(sim_matrix) - 1, len(A.T)))
    ax.set_yticklabels(["C%d" % i for i in range(len(A.T))])


def compare_cosine_similarity(sigma_prior_value, mu_value, num_component, dataset, save=False):
    fig = plt.figure(figsize=(8, 2))
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    cls_name = ['2-propanol', 'Ethanol', 'Methanol', '1-propanol']
    for i, s_sigma in enumerate(sigma_prior_value):
        for j, s_mu in enumerate(mu_value):
            if s_sigma == 0:
                method = ["infinity", "infinity"]
            else:
                method = ["limited", "limited"]
            tds_dir = get_model_dir(False, dataset, method, [s_sigma, s_sigma], [s_mu, s_mu],
                                    10000, num_component, 0)
            ax = fig.add_subplot(len(sigma_prior_value), len(mu_value),  i * len(mu_value) + j + 1)
            calc_cosine_similarity(tds_dir, ax, cls_name)
            # ax.set_title(r'$\Sigma_a$' + "=%d " % s_sigma + r'$\Sigma_b$' + "=%d " % s_sigma + \
            #              r'$\mu_a$' + "=%d " % s_mu + r'$\mu_b$' + "=%d" % s_mu, fontsize=10)
    if len(sigma_prior_value) > 1 or len(mu_value) > 1:
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax_global.set_xlabel("\n\n Reference spectra (pure component)", fontsize=10)
    ax_global.set_ylabel("Extracted component \n\n", fontsize=10)
    if save:
        plt.savefig(tds_dir + "/cosine_similarity_between_component_and_reference.pdf", pad_inches=0,
                    bbox_inches='tight')


def visualize_mu_convergence(mean_cov, skip=10,
                             tds_dir=None, save=False):
    fig = plt.figure(figsize=(10, 8))
    for str_index, str_use in enumerate(["A_mu", "B_mu"]):
        value = mean_cov[str_use][-5000:]
        if str_use == "A_mu":
            value = np.transpose(value, (0, 2, 1))  # num_samples, num_component, num_features
        print("The shape of the Mu matrix", np.shape(value))

        value_new = []
        for i, s_value in enumerate(value[::skip]):
            value_new.append(s_value)
            value_new.append(np.ones([2, np.shape(s_value)[1]]))
        value_new = np.concatenate(value_new, axis=0)
        ax = fig.add_subplot(2, 1, str_index + 1)
        ax.imshow(value_new)
        ax.set_yticks(np.linspace(0, len(value_new) - 1, len(value_new) // np.shape(value)[1] // 10))
        ax.set_yticklabels(np.linspace(5000, 10000, len(value_new) // np.shape(value)[1] // 10).astype('int32'))
        if str_use == "A_mu":
            ax.set_xlabel("Number of features (I)", fontsize=8)
        if str_use == "B_mu":
            ax.set_xlabel("Number of samples (J)", fontsize=8)
        if str_index == 0:
            ax.set_ylabel("Gibbs iterations", fontsize=8)
    fig.tight_layout()
    if save:
        plt.savefig(tds_dir + "/convergence_statistics_mu.pdf",
                    pad_inches=0, bbox_inches='tight')


def visualize_sigma_convergence(mean_cov, tds_dir, save=False):
    sigma_plot = mean_cov["A_sigma"][::10]
    ncol = 20
    nrow = len(sigma_plot) // ncol
    fig, axes = plt.subplots(nrows=nrow * 2 + 1, ncols=ncol, figsize=(8, nrow ))
    for i in range(nrow):
        for j in range(ncol):
            _index = i * ncol + j
            value_show = sigma_plot[_index]
            axes[i, j].imshow(value_show)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0 and j == ncol // 2:
                axes[i, j].set_title(r'$\Sigma_a$', fontsize=12)
    sigma_plot = mean_cov["B_sigma"][::10]
    for j in range(ncol):
        axes[nrow, j].axis('off')
        axes[nrow, j].set_xticks([])
        axes[nrow, j].set_yticks([])
    for i in range(nrow):
        for j in range(ncol):
            _index = i * ncol + j
            value_show = sigma_plot[_index]
            axes[i + nrow + 1, j].imshow(value_show)
            axes[i + nrow + 1, j].set_xticks([])
            axes[i + nrow + 1, j].set_yticks([])
            if i == 0 and j == ncol // 2:
                axes[i + nrow + 1, j].set_title(r'$\Sigma_b$', fontsize=12)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if save:
        plt.savefig(tds_dir + "/convergence_statistics_Sigma.pdf",
                    pad_inches=0, bbox_inches='tight')


def visualize_convergence_statistics(dataset, sigma_prior, sigma_prior_method, mu_prior, num_components,
                                     save=False):
    tds_dir = get_model_dir(False, dataset, [sigma_prior_method, sigma_prior_method],
                                  [sigma_prior, sigma_prior], [mu_prior, mu_prior], 10000, num_components,
                                  0)
    mean_cov = pickle.load(open(tds_dir + "mean_and_covariance_stat.obj", "rb"))
    for k in mean_cov.keys():
        mean_cov[k] = np.array(mean_cov[k])
        print(k, mean_cov[k].shape)
    # visualize_mu_convergence(mean_cov, 20, tds_dir=tds_dir, save=save)
    visualize_sigma_convergence(mean_cov, tds_dir, save=save)


def show_loss(MSE_error, num_samples, tds_dir, save=False):
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    x_value = np.arange(len(MSE_error))[100:]
    y_value = MSE_error[100:]
    ax.plot(x_value, y_value / num_samples, 'r')
    ax.set_xlabel("Gibbs steps", fontsize=10)
    if "sanitizer" in tds_dir:
        ax.set_ylabel("MSE per spectrum", fontsize=10)
    else:
        ax.set_ylabel("MSE per image", fontsize=10)
    if save:
        plt.savefig(tds_dir + "/MSE_error.pdf", pad_inches=0, bbox_inches='tight')


def visualize_component_concentration(tds_dir,  o_spectra_norm, cls_label, cls_name, ref_label, mix_ratio,
                                      data_type="simulate", save=False):
    A = np.load(tds_dir + "/A.npy")
    B = np.load(tds_dir + "/B.npy")
    cos_sim, component_label = get_label_for_each_component(A / np.max(A, axis=0, keepdims=True),
                                                                  o_spectra_norm, cls_label)
    print("predicted component label", component_label)
    B_pred = B.T
    mixture_cases_interest = [[0, 2], [0, 3], [1, 2], [1, 3]]
    title_group = ["+".join(cls_name[v]) for v in mixture_cases_interest]
    num_stat = []
    fig = plt.figure(figsize=(13, 3))
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    for i, s_mix_label in enumerate(mixture_cases_interest):
        ax = fig.add_subplot(1, 4, i+1)
        index = np.where(np.sum(ref_label == s_mix_label, axis=1) == 2)[0]
        if data_type == "simulate":
            index = index[np.sum(mix_ratio[index], axis=-1) == 1]
        ax.scatter(mix_ratio[index, 0], mix_ratio[index, 1], color='r')
        pred_conc = B_pred[index, :]
        pred_label = component_label[np.argsort(pred_conc, axis=1)[:, -2:]]
        s_stat = []
        for j, k in enumerate(mixture_cases_interest):
            _pred_num = len(np.where(np.sum(np.sort(pred_label, axis=-1) == k, axis=1) == 2)[0])
            s_stat.append(_pred_num)
        num_stat.append(s_stat)
        ax.scatter(B[s_mix_label[0], index], B[s_mix_label[1], index], color='g')
        ax.legend(["Measured concentration", "Predicted concentration"], fontsize=8)
        ax.set_title(title_group[i], fontsize=8)
    plt.subplots_adjust(wspace=0.2)
    ax_global.set_xlabel("\n\n1st component concentration", fontsize=8)
    ax_global.set_ylabel("2nd component concentration\n\n", fontsize=8)
    if save:
        plt.savefig(tds_dir + "/measure_vs_predicted_concentration_comp.pdf",
                    pad_inches=0, bbox_inches='tight')
    title_group = ["+\n".join(cls_name[v]) for v in mixture_cases_interest]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    sns.heatmap(num_stat, annot=True, ax=ax, cbar=False, cmap='YlGnBu', fmt='0.0f',
                xticklabels=title_group, yticklabels=title_group)
    if save:
        plt.savefig(tds_dir + "/measure_vs_predicted_concentration_heatmap.pdf",
                    pad_inches=0, bbox_inches='tight')


def visualize_qualitative_result(s_method, s_sigma, s_mu, num_component, data_dir, save=False):
    tds_dir = get_model_dir(False, "sanitizer", [s_method, s_method], [s_sigma, s_sigma], [s_mu, s_mu],
                            10000, num_component, 0)
    alcohol = pickle.load(open(tds_dir + "/alcohol_data.obj", "rb"))
    one_component_stat, two_component_stat, _, _ = load_sanitizer.get_alcohol_data(data_dir)
    o_spectra = alcohol["one_component_spectra"]
    manu_mix_ratio = alcohol["ratio"]
    cls_name = np.array(['2-propanol', 'Ethanol', 'Methanol', '1-propanol'])
    o_spectra_norm = o_spectra / np.max(o_spectra, axis=1, keepdims=True)
    o_label = alcohol["mixture_label"][0]
    one_comp_cls_label = np.argmax(one_component_stat[-1], axis=-1)
    visualize_component_concentration(tds_dir, o_spectra_norm, one_comp_cls_label,
                                      cls_name, o_label, manu_mix_ratio,
                                      data_type="simulate", save=save)


def show_example_and_mixture_spectra(data_dir="/Users/blia/Documents/experiments/mixture_data/",
                                     tds_dir=None, save=False):
    one_component_stat, two_component_stat, component_class, wavenumbers = load_sanitizer.get_alcohol_data(data_dir)
    spectra, _, concentration = one_component_stat
    spectra = spectra / np.max(spectra, axis=-1, keepdims=True)
    one_label = np.argmax(concentration, axis=-1)
    avg_spec = [np.mean(spectra[np.where(one_label == i)[0]], axis=0) for i in range(4)]
    color_group = ['r', 'g', 'b', 'orange', "m"]
    fig = plt.figure(figsize=(8, 4))
    ax_global = ax_global_get(fig)
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax = fig.add_subplot(211)
    for i, s_spec in enumerate(avg_spec):
        ax.plot(wavenumbers, s_spec, color=color_group[i], label=component_class[i])
    ax.legend(loc='best', fontsize=9)
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax = fig.add_subplot(212)
    ratio = np.linspace(0, 1, 6)[1:-1]
    ref_index = 1
    toxic_index = 2
    mixture_0 = [avg_spec[ref_index] * ratio[i] + avg_spec[toxic_index] * (1 - ratio[i]) for i in range(len(ratio))]
    for i, s_spec in enumerate(mixture_0):
        la_0 = "%s: %.2f" % (component_class[ref_index], ratio[i])
        la_1 = "%s: %.2f" % (component_class[toxic_index], 1 - ratio[i])
        la = "+".join([la_0, la_1])
        ax.plot(wavenumbers, s_spec, color=color_group[i], label=la)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend(loc='best', fontsize=9)
    ax_global.set_xlabel("\n\nWavenumber(cm" + r'$^{-1}$' + ")", fontsize=10)
    ax_global.set_ylabel("A.U.\n\n", fontsize=10)
    plt.subplots_adjust(hspace=0.1)
    if save:
        plt.savefig(tds_dir + "/example_pure_spectra_and_mixtures.pdf", pad_inches=0, bbox_inches='tight')


def show_mnist(x_train, y_train, mixtures, tds_dir, save):
    fig = plt.figure(figsize=(10, 3))
    original_image = []
    for i, i_la in enumerate(np.unique(y_train)):
        subset = np.where(y_train == i_la)[0]
        subindex = np.random.choice(subset, 3, replace=False)
        original_image.append(np.reshape(x_train[subindex], [-1, 28, 28]))
    original_image = np.transpose(original_image, (1, 0, 2, 3))
    canvas_im = create_canvas(original_image)
    select_index = np.random.choice(np.arange(len(mixtures)), 30, replace=False)
    mixture_canvas = create_canvas(np.reshape(mixtures[select_index], [3, 10, 28, 28]))
    title_use = ["original image", "mixed image"]
    for i, s_im in enumerate([canvas_im, mixture_canvas]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(s_im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title_use[i])
    plt.subplots_adjust(wspace=0.1)
    if save:
        plt.savefig(tds_dir + "/mnist_example_images.pdf", pad_inches=0, bbox_inches='tight')


def visualize_component_matrix_mnist(sigma_prior, sigma_prior_method, mu_prior, num_component, save=False):
    tds_dir = get_model_dir(False, "mnist", [sigma_prior_method, sigma_prior_method],
                                  [sigma_prior, sigma_prior],
                                  [mu_prior, mu_prior],
                                  10000, num_component, version=0)
    A = np.load(tds_dir + "A.npy")
    error = np.load(tds_dir + "Error.npy") / 4000
    num_feat, num_comp = A.shape
    A_feat = np.reshape(A.T, [num_comp // 10, 10, 28, 28])
    ca = create_canvas(A_feat)
    fig = plt.figure(figsize=(8, 1.3))
    ax = fig.add_subplot(121)
    ax.plot(np.arange(len(error))[500:], error[500:], 'r')
    ax.set_xlabel("Gibbs step", fontsize=10)
    ax.set_ylabel("MSE", fontsize=10)
    ax = fig.add_subplot(122)
    ax.imshow(ca)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(wspace=0.05)
    if save:
        plt.savefig(tds_dir + "/extracted_components.pdf", pad_inches=0, bbox_inches='tight')


def show_worst_predictions(sigma_prior, sigma_prior_method, mu_prior, num_component, mixture,
                           save=False):
    tds_dir = get_model_dir(False, "mnist", [sigma_prior_method, sigma_prior_method],
                                  [sigma_prior, sigma_prior],
                                  [mu_prior, mu_prior],
                                  10000, num_component, version=0)
    A = np.load(tds_dir + "A.npy")
    B = np.load(tds_dir + "B.npy")
    prediction = np.matmul(A, B)
    cos_sim = cosine_similarity(prediction.T,
                                mixture.T)[np.arange(len(mixture)),
                                           np.arange(len(mixture))]  # [num_prediction, num_samples]
    worst_20 = np.argsort(cos_sim)[:20]
    s_index = worst_20
    method = ["input image", "prediction"]
    fig = plt.figure(figsize=(8, 3))
    cos_ca_original = create_canvas(np.reshape(mixture[:, s_index].T, [2, 10, 28, 28]))
    cos_ca_pred = create_canvas(np.reshape(prediction[:, s_index].T, [2, 10, 28, 28]))
    for i, s_ca in enumerate([cos_ca_original, cos_ca_pred]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(s_ca)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(method[i], fontsize=10)
    plt.subplots_adjust(wspace=0.1)
    if save:
        plt.savefig(tds_dir + "/less_similar_predictions.pdf", pad_inches=0, bbox_inches='tight')


def enlarge_visualize(heatmap, factor):
    """
    Args:
        heatmap: [Latent dimension, Wavenumber]
        factor: int
    """
    new_map = []
    num_z, num_wave = np.shape(heatmap)
    for i in range(num_z):
        _map = np.repeat(heatmap[i:(i + 1), :], factor, axis=0)
        new_map.append(_map)
        # if i != num_z - 1:
        #     new_map.append(np.ones([2, num_wave]))
    new_map = np.array([v for j in new_map for v in j])
    return new_map


def create_canvas(feature):
    """This function is used to create canvas
    Args:
        feature [out_channel, in_channel, kh, kw]
    """
    nx, ny, fh, fw = np.shape(feature)
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.empty((fw * nx, fh * ny))
    for i, yi in enumerate(x_values):
        f_sub = feature[i]
        for j, xj in enumerate(y_values):
            f_use = f_sub[j]
            canvas[(nx - i - 1) * fh:(nx - i) * fh,
            j * fw:(j + 1) * fw] = f_use
    return canvas


def ax_global_get(fig):
    ax_global = fig.add_subplot(111, frameon=False)
    ax_global.spines['top'].set_color('none')
    ax_global.spines['bottom'].set_color('none')
    ax_global.spines['left'].set_color('none')
    ax_global.spines['right'].set_color('none')
    ax_global.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    return ax_global
