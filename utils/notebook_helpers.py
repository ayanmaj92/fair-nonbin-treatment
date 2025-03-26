import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import numpy as np
import os


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def load_csv_file(root_path, data_name, split):
    assert split in ['train', 'val', 'test', 'all']
    if split in ['train', 'val', 'test']:
        result_path = os.path.join(f'treatment_{data_name}_dummy', 'evaluate', 'pscf_analysis', f'{split}.csv')
        return pd.read_csv(os.path.join(root_path, result_path), header=0)
    else:
        list_dfs = []
        for split in ['train', 'val', 'test']:
            result_path = os.path.join(f'treatment_{data_name}_dummy', 'evaluate', 'pscf_analysis', f'{split}.csv')
            list_dfs.append(pd.read_csv(os.path.join(root_path, result_path), header=0))
        return pd.concat(list_dfs, ignore_index=True)


def load_yaml_config(root_path, data_name):
    params = None
    yaml_path = os.path.join(f'treatment_{data_name}_dummy', 'evaluate', 'wandb_local', 'config.yaml')
    with open(os.path.join(root_path, yaml_path)) as stream:
        params = yaml.safe_load(stream)
    return params


def dtd_itd_analysis(df, params, plot=False, plot_params=None):
    treatment_col_idxs = range(params['dataset__num_sensitive'] + params['dataset__num_covariate'],
                               params['dataset__num_sensitive'] + params['dataset__num_covariate'] + params[
                                   'dataset__num_treatment'])

    treatment_features = [c.split('_F')[0] for c in df.columns[treatment_col_idxs]]

    gender_col_name = df.columns[0].split('_F')[0]

    # Overall
    treatment_vals_F = df[[f'{c}_F' for c in treatment_features]].to_numpy()
    treatment_vals_CF = df[[f'{c}_CF' for c in treatment_features]].to_numpy()

    treatment_vals_PSDT = df[[f'{c}_PSDT' for c in treatment_features]].to_numpy()

    td_vals = treatment_vals_CF - treatment_vals_F
    td_mean = np.median(td_vals, axis=0)
    td_std = np.std(td_vals, axis=0)

    new_dtd_vals = treatment_vals_PSDT - treatment_vals_F
    new_dtd_mean = np.median(new_dtd_vals, axis=0)
    new_dtd_std = np.std(new_dtd_vals, axis=0)

    td_dict = {c: [v1, v2] for c, v1, v2 in zip(treatment_features, td_mean, td_std)}
    new_dtd_dict = {c: [v1, v2] for c, v1, v2 in zip(treatment_features, new_dtd_mean, new_dtd_std)}

    results_dict = {'Overall': {'TD': td_dict, 'DTD': new_dtd_dict}}

    # For Group 0
    treatment_vals_F_0 = df[df[f'{gender_col_name}_F'] == 0][
        [f'{c}_F' for c in treatment_features]].to_numpy()
    treatment_vals_CF_0 = df[df[f'{gender_col_name}_F'] == 0][
        [f'{c}_CF' for c in treatment_features]].to_numpy()
    treatment_vals_PSDT_0 = df[df[f'{gender_col_name}_F'] == 0][
        [f'{c}_PSDT' for c in treatment_features]].to_numpy()

    td_vals_0 = treatment_vals_CF_0 - treatment_vals_F_0
    td_mean_0 = np.median(td_vals_0, axis=0)
    td_std_0 = np.std(td_vals_0, axis=0)

    new_dtd_vals_0 = treatment_vals_PSDT_0 - treatment_vals_F_0
    new_dtd_mean_0 = np.median(new_dtd_vals_0, axis=0)
    new_dtd_std_0 = np.std(new_dtd_vals_0, axis=0)

    td_dict_0 = {c: [v1, v2] for c, v1, v2 in zip(treatment_features, td_mean_0, td_std_0)}
    new_dtd_dict_0 = {c: [v1, v2] for c, v1, v2 in
                      zip(treatment_features, new_dtd_mean_0, new_dtd_std_0)}

    results_dict[f'{gender_col_name} 0'] = {'TD': td_dict_0, 'DTD': new_dtd_dict_0}

    # For Group 1
    treatment_vals_F_1 = df[df[f'{gender_col_name}_F'] == 1][
        [f'{c}_F' for c in treatment_features]].to_numpy()
    treatment_vals_CF_1 = df[df[f'{gender_col_name}_F'] == 1][
        [f'{c}_CF' for c in treatment_features]].to_numpy()
    treatment_vals_PSDT_1 = df[df[f'{gender_col_name}_F'] == 1][
        [f'{c}_PSDT' for c in treatment_features]].to_numpy()

    td_vals_1 = treatment_vals_CF_1 - treatment_vals_F_1
    td_mean_1 = np.median(td_vals_1, axis=0)
    td_std_1 = np.std(td_vals_1, axis=0)

    new_dtd_vals_1 = treatment_vals_PSDT_1 - treatment_vals_F_1
    new_dtd_mean_1 = np.median(new_dtd_vals_1, axis=0)
    new_dtd_std_1 = np.std(new_dtd_vals_1, axis=0)

    td_dict_1 = {c: [v1, v2] for c, v1, v2 in zip(treatment_features, td_mean_1, td_std_1)}
    new_dtd_dict_1 = {c: [v1, v2] for c, v1, v2 in
                      zip(treatment_features, new_dtd_mean_1, new_dtd_std_1)}

    results_dict[f'{gender_col_name} 1'] = {'TD': td_dict_1, 'DTD': new_dtd_dict_1}

    if plot:
        for tag, dat, dat0, dat1 in \
                [('F', treatment_vals_F, treatment_vals_F_0, treatment_vals_F_1),
                 ('$S^{CF}$', treatment_vals_CF, treatment_vals_CF_0, treatment_vals_CF_1),
                 ('$S^{CF}, X^F$', treatment_vals_PSDT, treatment_vals_PSDT_0, treatment_vals_PSDT_1)]:
            fig, axes = plt.subplots(nrows=plot_params['rows'], ncols=plot_params['cols'],
                                     figsize=plot_params['figsize'])
            for i, ax in enumerate(axes.flatten()):
                sns.kdeplot(ax=ax, data=dat[:, i], label='ovr')
                sns.kdeplot(ax=ax, data=dat0[:, i], label='$S_{F}0$')
                sns.kdeplot(ax=ax, data=dat1[:, i], label='$S_{F}1$')
                ax.set_xlabel(treatment_features[i])
                ax.legend()
                if i != 0:
                    ax.set_ylabel('')
                if i != plot_params['plot_idx_legend']:
                    ax.legend().remove()
            fig.suptitle(f'Treatment({tag})')
            plt.show()

        for tag, dat, dat0, dat1 in \
                [('TD $Z^{CF}-Z^F$', td_vals, td_vals_0, td_vals_1),
                 ('DTD $Z(S^{CF},X^F)-Z^F$', new_dtd_vals, new_dtd_vals_0, new_dtd_vals_1)]:
            fig, axes = plt.subplots(nrows=plot_params['rows'], ncols=plot_params['cols'],
                                     figsize=plot_params['figsize'])

            for i, ax in enumerate(axes.flatten()):
                sns.kdeplot(ax=ax, data=dat[:, i], label='ovr')
                sns.kdeplot(ax=ax, data=dat0[:, i], label='$S_{F}0$')
                sns.kdeplot(ax=ax, data=dat1[:, i], label='$S_{F}1$')
                ax.set_xlabel(treatment_features[i])
                ax.legend()
                if i != 0:
                    ax.set_ylabel('')
                if i != plot_params['plot_idx_legend']:
                    ax.legend().remove()
            fig.suptitle(f'Discr. {tag}')
            plt.show()
    return results_dict
