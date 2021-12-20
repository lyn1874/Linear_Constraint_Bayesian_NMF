import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def get_spectra(num_spectra, wavenumbers_string, data_frame, norm=None):
    spectra = np.zeros([num_spectra, len(wavenumbers_string)])
    for i, v in enumerate(wavenumbers_string):
        spectra[:, i] = data_frame[v].to_numpy()
    if norm == "l1":
        spectra = spectra / np.sum(spectra, axis=-1, keepdims=True)
    return spectra


def produce_training_pure_mixture_figures(data_path, data_files, tds_dir, norm=None, show=False, save=False):
    conc_tr = [v for v in data_files if "Training" in v and "Concentration" in v][0]
    raman_tr = [v for v in data_files if "Figure4" in v and "Training_RamanData" in v][0]
    conc_tr_data = pd.read_csv(data_path + conc_tr)
    raman_tr_data = pd.read_csv(data_path + raman_tr, header=1)
    container_class = raman_tr_data["Unnamed: 0"].to_numpy()
    component_class = list(conc_tr_data.keys())
    component_conc = conc_tr_data.to_numpy()
    num_components = np.array([len(np.where(v != 0)[0]) for v in component_conc])
    wavenumbers = np.array([int(v) for v in list(raman_tr_data.keys())[4:]])
    spectra = get_spectra(len(container_class), list(raman_tr_data.keys())[4:], raman_tr_data,
                          norm=norm)
    print("There are %d spectra in total" % len(spectra))
    two_component_index = np.where(num_components > 1)[0]
    print("%d of them are mixtures" % len(two_component_index))
    print("%d of them only contain single alcohol" % (len(spectra) - len(two_component_index)))
    non_zero_index = np.array([np.where(v != 0)[0] for v in component_conc[two_component_index, :]])
    color_use = ['r', 'g', 'b', 'orange', 'm']
    if show:
        for i in np.unique(non_zero_index[:, 0]):
            fig = plt.figure(figsize=(10, 3))
            sub_index = np.where(non_zero_index[:, 0] == i)[0]
            sub_spectra = spectra[two_component_index][sub_index]
            sub_conc = component_conc[two_component_index][sub_index]
            for j_index, j in enumerate(np.unique(non_zero_index[:, 1])):
                ax = fig.add_subplot(1, 2, j_index+1)
                sub_component_class = [component_class[i], component_class[j]]
                _sub_index = np.where(non_zero_index[sub_index, 1] == j)[0]
                _sub_spectra = sub_spectra[_sub_index]
                _conc = sub_conc[_sub_index]
                for m, _s_spec in enumerate(_sub_spectra):
                    _sub_conc_legend = [_conc[m][i], _conc[m][j]]
                    _container_label = container_class[sub_index][_sub_index][m]
                    _label_use = _container_label + " " + ' '.join(["%s:%d" % (v, q) for v, q in zip(sub_component_class,
                                                                                                     _sub_conc_legend)])
                    ax.plot(wavenumbers, _s_spec, color=color_use[m], label=_label_use)
                ax.legend(loc='best', fontsize=8, handlelength=0.5)
            if save:
                plt.savefig(tds_dir + "/%s_contaminated.pdf" % (component_class[i]), pad_inches=0, bbox_inches='tight')
    one_component_index = np.where(num_components == 1)[0]
    non_zero_index = np.array([np.where(v != 0)[0][0] for v in component_conc[one_component_index, :]])
    if show:
        for i in np.unique(non_zero_index):
            fig = plt.figure(figsize=(11, 3))
            _index = np.where(non_zero_index == i)[0]
            _sub_spec = spectra[one_component_index][_index]
            _sub_conc = component_conc[one_component_index][_index][:, i]
            _sub_container = container_class[one_component_index][_index]
            for j, _s_container in enumerate(np.unique(_sub_container)):
                ax = fig.add_subplot(1, len(np.unique(_sub_container)), j+1)
                _select = np.where(_sub_container == _s_container)[0]
                _select = _select[np.argsort(_sub_conc[_select])]
                _sub_sub_con = _sub_conc[_select]
                for m, _s_spec in enumerate(_sub_spec[_select]):
                    ax.plot(wavenumbers, _s_spec, label="%d" % (_sub_sub_con[m]))
                ax.set_title("%s (%s)" % (component_class[i], _s_container))
                ax.legend(loc='best', handlelength=0.5)
            plt.subplots_adjust(wspace=0.25)
            if save:
                plt.savefig(tds_dir + "/pure_alcohol_%s.pdf" % (component_class[i]), pad_inches=0, bbox_inches='tight')

    one_component_statistics = [spectra[one_component_index][:, :-1], container_class[one_component_index],
                                component_conc[one_component_index]]
    two_component_statistics = [spectra[two_component_index][:, :-1], container_class[two_component_index],
                                component_conc[two_component_index]]
    return one_component_statistics, two_component_statistics, component_class, wavenumbers[:-1]


def get_alcohol_data(data_path, normalisation="l1"):
    # data_path = "../mixture_data/"
    data_files = sorted([v for v in os.listdir(data_path) if '.csv' in v and "._" not in v])
    one_component_statistics, \
        two_component_statistics, \
        component_class, wavenumbers = produce_training_pure_mixture_figures(data_path,
                                                                                   data_files,
                                                                                   None,
                                                                                   norm=normalisation,
                                                                                   show=False,
                                                                                   save=False)
    return one_component_statistics, two_component_statistics, component_class, wavenumbers


def prepare_mixture_data(tds_dir):
    if os.path.exists("../mixture_data/"):
        dir2load_data = "../mixture_data/"
    else:
        dir2load_data = "/Users/bo/Documents/experiments/mixture_data/"

    one_component, two_component, component_cls, wavenumber = get_alcohol_data(dir2load_data)
    o_spectra, o_container, o_concentration = one_component
    o_cls_label = np.argmax(o_concentration, axis=-1)
    t_spectra, t_container, t_concentration = two_component
    t_cls_label = np.argsort(t_concentration, axis=-1)[:, -2:]
    rand_index = [np.random.choice(len(o_spectra), len(o_spectra), replace=False) for _ in range(20)]
    rand_index = np.concatenate(rand_index, axis=0)
    ratio = np.random.rand(len(rand_index), 2)
    ratio = ratio/np.sum(ratio, axis=-1, keepdims=True)
    ratio[ratio < 0.2] = 0.2

    orig_spectra = np.concatenate([o_spectra for _ in range(20)], axis=0)
    mix_spectra = orig_spectra[rand_index]
    target_spectra = orig_spectra * ratio[:, 0:1] + mix_spectra * ratio[:, 1:2]
    print(np.shape(orig_spectra), np.shape(mix_spectra), np.shape(target_spectra))

    orig_label = np.concatenate([o_cls_label for _ in range(20)], axis=0)
    mix_label = orig_label[rand_index]
    targ_label = np.hstack([np.expand_dims(orig_label, axis=1), np.expand_dims(mix_label, axis=1)])

    combined_mixture = np.concatenate([target_spectra, two_component[0]], axis=0)

    data = {}
    data["mixture_data"] = np.transpose(combined_mixture, (1, 0))
    data["mixture_label"] = [targ_label, t_cls_label]
    data["ratio"] = ratio
    data["mix_index"] = rand_index
    data["one_component_spectra"] = one_component[0]
    data["two_component_spectra"] = two_component[0]
    data["wavenumber"] = wavenumber
    with open(tds_dir + "/alcohol_data.obj", 'wb') as f:
        pickle.dump(data, f)

    return np.transpose(combined_mixture, (1, 0)), [targ_label,  t_cls_label], ratio
