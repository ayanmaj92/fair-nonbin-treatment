import torch

from causal_flows.zuko.transforms import ComposedTransform


def get_sensitive_counterfactual_mask(sensitive_vals, floored=True):
    """Return interventional S and interventional S with noise to be used in Flows"""
    orig_s_vals = torch.clone(sensitive_vals)
    if not floored:
        orig_s_vals = torch.floor(orig_s_vals)
    intervention_s_vals = torch.zeros_like(orig_s_vals)
    intervention_s_vals[orig_s_vals == 0] = 1
    intervention_s_vals += 0.5
    return intervention_s_vals


def compute_sensitive_counterfactual(data, model, prep, sensitive_idx=0):
    sens_vals = data[:, sensitive_idx]
    sens_val_cf = get_sensitive_counterfactual_mask(sens_vals, floored=False)
    data_sens_cf = model.model.compute_counterfactual(data, sensitive_idx, sens_val_cf,
                                                      scaler=prep.scaler_transform, return_dict=False)
    return data_sens_cf


def compute_path_specific_downstream_effect_treatment_intervention(data, model, prep, cf_data):
    treatment_idxs = list(range(prep.num_sensitive + prep.num_covariate,
                                prep.num_sensitive + prep.num_covariate + prep.num_treatment))
    max_col_vals = torch.max(prep.datasets[0].x, dim=0).values
    min_col_vals = torch.min(prep.datasets[0].x, dim=0).values
    idxs = [idx for idx in treatment_idxs if idx in prep.binary_dims]
    cf_copy = torch.clone(cf_data)
    cf_copy[:, idxs] = torch.clamp(cf_copy[:, idxs],
                                   min=min_col_vals[idxs],
                                   max=max_col_vals[idxs])
    cf_tmp = torch.clone(cf_copy)
    cf_tmp[:, idxs] = torch.floor(cf_tmp[:, idxs]) + 0.5

    treatment_vals = cf_tmp[:, treatment_idxs]
    treat_cf = model.model.compute_counterfactual(data, treatment_idxs, treatment_vals,
                                                  scaler=prep.scaler_transform, return_dict=False)
    return treat_cf


@torch.no_grad()
def compute_path_specific_sensitive_direct_label(data, model, prep, sens_cf_data):
    scaler = prep.scaler_transform
    n_flow = model.model.flow()
    if scaler is not None:
        n_flow.transform = ComposedTransform(scaler, n_flow.transform)
    u_factual = n_flow.transform(data)
    u_f_y = u_factual[:, -1].view(-1, 1)
    u_cf_pass = n_flow.transform(sens_cf_data)
    u_cf_s = u_cf_pass[:, :prep.num_sensitive]
    if prep.num_sensitive == 1:
        u_cf_s = u_cf_s.view(-1, 1)
    intervention_idxs = list(range(prep.num_sensitive, prep.num_sensitive + prep.num_covariate + prep.num_treatment))
    intervention_vals = torch.clone(data)
    idxs = [idx for idx in intervention_idxs if idx in prep.binary_dims]
    max_col_vals = torch.max(prep.datasets[0].x, dim=0).values
    min_col_vals = torch.min(prep.datasets[0].x, dim=0).values
    intervention_vals[:, idxs] = torch.clamp(intervention_vals[:, idxs],
                                             min=min_col_vals[idxs],
                                             max=max_col_vals[idxs])
    intervention_vals[:, idxs] = torch.floor(intervention_vals[:, idxs]) + 0.5
    interim_data = torch.clone(sens_cf_data)
    interim_data[:, intervention_idxs] = intervention_vals[:, intervention_idxs]
    u_intv = n_flow.transform(interim_data)
    u_x_z_f_intv = u_intv[:, prep.num_sensitive:prep.num_sensitive + prep.num_covariate + prep.num_treatment]
    u_combined = torch.cat((u_cf_s, u_x_z_f_intv, u_f_y), dim=1)
    cf_entity = n_flow.transform.inv(u_combined)
    return cf_entity


@torch.no_grad()
def compute_path_specific_treatment(data, model, prep, sens_cf_data):
    scaler = prep.scaler_transform
    n_flow = model.model.flow()
    if scaler is not None:
        n_flow.transform = ComposedTransform(scaler, n_flow.transform)
    # 1. Get U^factual
    u_factual = n_flow.transform(data)
    u_f_y = u_factual[:, -1].view(-1, 1)
    u_f_z = u_factual[:, prep.num_sensitive + prep.num_covariate:prep.num_sensitive + prep.num_covariate + prep.num_treatment]
    if prep.num_treatment == 1:
        u_f_z = u_f_z.view(-1, 1)
    # 2. do(S->S^CF) to get U_S^CF
    u_cf_pass = n_flow.transform(sens_cf_data)
    u_cf_s = u_cf_pass[:, :prep.num_sensitive]
    # 3. do(X^F) with S^CF to get U_X^F
    intervention_idxs = list(range(prep.num_sensitive, prep.num_sensitive + prep.num_covariate))
    intervention_vals = torch.clone(data)
    max_col_vals = torch.max(prep.datasets[0].x, dim=0).values
    min_col_vals = torch.min(prep.datasets[0].x, dim=0).values
    idxs = [idx for idx in intervention_idxs if idx in prep.binary_dims]
    intervention_vals[:, idxs] = torch.clamp(intervention_vals[:, idxs],
                                             min=min_col_vals[idxs],
                                             max=max_col_vals[idxs])
    intervention_vals[:, idxs] = torch.floor(intervention_vals[:, idxs]) + 0.5
    interim_data = torch.clone(sens_cf_data)
    interim_data[:, intervention_idxs] = intervention_vals[:, intervention_idxs]
    u_intv = n_flow.transform(interim_data)
    u_x_f_intv = u_intv[:, prep.num_sensitive:prep.num_sensitive + prep.num_covariate]
    if prep.num_covariate == 1:
        u_x_f_intv = u_x_f_intv.view(-1, 1)
    # 4. Get Z^PSCF(do(S^CF), do(X^F))
    u_combined = torch.cat((u_cf_s, u_x_f_intv, u_f_z, u_f_y), dim=1)
    cf_entity = n_flow.transform.inv(u_combined)
    return cf_entity


@torch.no_grad()
def compute_marginal_risk_score(data, model, prep, cf_samples):
    y_for_risk = torch.zeros((data.size(0), cf_samples.size(0)))
    max_col_vals = torch.max(prep.datasets[0].x, dim=0).values
    min_col_vals = torch.min(prep.datasets[0].x, dim=0).values
    for i, cf_sample in enumerate(cf_samples):
        cf_batch = cf_sample.repeat(data.size(0), cf_samples.size(1))
        treat_cf = compute_path_specific_downstream_effect_treatment_intervention(data, model, prep, cf_batch)
        treat_cf_data = torch.clone(treat_cf)
        treat_cf_data[:, -1] = torch.clamp(treat_cf_data[:, -1], min=min_col_vals[-1], max=max_col_vals[-1])
        treat_cf_data[:, -1] = torch.floor(treat_cf_data[:, -1])
        y_for_risk[:, i] = treat_cf_data[:, -1]
    return 1.0 - torch.mean(y_for_risk, dim=1)
