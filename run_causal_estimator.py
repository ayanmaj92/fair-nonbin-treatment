import glob
import pickle
import pandas as pd
from pathlib import Path
import wandb
import os
import causal_flows.causal_nf.config as causal_nf_config
import causal_flows.causal_nf.utils.training as causal_nf_train
import causal_flows.causal_nf.utils.wandb_local as wandb_local
from causal_flows.causal_nf.config import cfg
import causal_flows.causal_nf.utils.io as causal_nf_io
import warnings
from utils.helpers import *

warnings.simplefilter("ignore")

os.environ["WANDB_NOTEBOOK_NAME"] = "name_of_the_notebook"

args_list, args = causal_nf_config.parse_args()

load_model = isinstance(args.load_model, str)
if load_model:
    causal_nf_io.print_info(f"Loading model: {args.load_model}")

config = causal_nf_config.build_config(
    config_file=args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)

causal_nf_config.assert_cfg_and_config(cfg, config)

if cfg.device in ["cpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
causal_nf_train.set_reproducibility(cfg)

print(cfg.dataset.name)

if "treatment" in cfg.dataset.name:
    from causal_flows.causal_nf.preparators.treatment_data_preparator import TreatmentDataPreparator

    preparator = TreatmentDataPreparator.loader(cfg.dataset)
elif cfg.dataset.name in ["german"]:
    from causal_flows.causal_nf.preparators.german_preparator import GermanPreparator

    preparator = GermanPreparator.loader(cfg.dataset)
else:
    from causal_flows.causal_nf.preparators.scm import SCMPreparator

    preparator = SCMPreparator.loader(cfg.dataset)

preparator.prepare_data()

loaders = preparator.get_dataloaders(
    batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
)

for i, loader in enumerate(loaders):
    causal_nf_io.print_info(f"[{i}] num_batches: {len(loader)}")

model = causal_nf_train.load_model(cfg=cfg, preparator=preparator)

param_count = model.param_count()
config["param_count"] = param_count

if not load_model:
    assert isinstance(args.project, str)
    run = wandb.init(
        mode=args.wandb_mode,
        group=args.wandb_group,
        project=args.project,
        config=config,
    )

    import uuid

    if args.wandb_mode != "disabled":
        run_uuid = run.id
    else:
        run_uuid = str(uuid.uuid1()).replace("-", "")
else:
    run_uuid = os.path.basename(args.load_model)

if 'test' in args.project.lower() or 'hparam' in args.project.lower():
    dirpath = os.path.join(cfg.root_dir, run_uuid)
else:
    dirpath = os.path.join(cfg.root_dir, f'{cfg.dataset.name}_{cfg.dataset.sem_name}')

if load_model:
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if 'test' in args.project.lower() or 'hparam' in args.project.lower():
        logger_dir = os.path.join(cfg.root_dir, run_uuid, "evaluate", now)
    else:
        logger_dir = os.path.join(cfg.root_dir, f'{cfg.dataset.name}_{cfg.dataset.sem_name}', "evaluate")
        # now)
else:
    if 'test' in args.project.lower() or 'hparam' in args.project.lower():
        logger_dir = os.path.join(cfg.root_dir, run_uuid)
    else:
        logger_dir = os.path.join(cfg.root_dir, f'{cfg.dataset.name}_{cfg.dataset.sem_name}')

trainer, logger = causal_nf_train.load_trainer(
    cfg=cfg,
    dirpath=dirpath,
    logger_dir=logger_dir,
    include_logger=True,
    model_checkpoint=cfg.train.model_checkpoint,
    cfg_early=cfg.early_stopping,
    preparator=preparator,
)

causal_nf_io.print_info(f"Experiment folder: {logger.save_dir}\n\n")

wandb_local.log_config(dict(config), root=logger.save_dir)

if not load_model:
    wandb_local.copy_config(
        config_default=causal_nf_config.DEFAULT_CONFIG_FILE,
        config_experiment=args.config_file,
        root=logger.save_dir,
    )
    trainer.fit(model, train_dataloaders=loaders[0], val_dataloaders=loaders[1])

if isinstance(preparator.single_split, str):
    loaders = [loaders[0]]

model.save_dir = dirpath

with open(os.path.join(model.save_dir, 'params_causal_model.pkl'), 'wb') as f:
    pickle.dump(cfg, f)

# This is where the model checkpoints are loaded and then tested!
if load_model:
    ckpt_name_list = glob.glob(os.path.join(args.load_model, f"last.ckpt"))
    for ckpt_file in ckpt_name_list:
        model = causal_nf_train.load_model(
            cfg=cfg, preparator=preparator, ckpt_file=ckpt_file
        )
        model.eval()
        model.save_dir = dirpath
        ckpt_name = preparator.get_ckpt_name(ckpt_file)
        for i, loader_i in enumerate(loaders):
            s_name = preparator.split_names[i]
            causal_nf_io.print_info(f"Testing {s_name} split")
            preparator.set_current_split(i)
            model.ckpt_name = ckpt_name
            _ = trainer.test(model=model, dataloaders=loader_i)
            metrics_stats = model.metrics_stats
            metrics_stats["current_epoch"] = trainer.current_epoch
            wandb_local.log_v2(
                {s_name: metrics_stats, "epoch": ckpt_name},
                root=trainer.logger.save_dir,
            )
else:
    ckpt_name_list = ["last"]
    if cfg.early_stopping.activate:
        ckpt_name_list.append("best")
    for ckpt_name in ckpt_name_list:
        for i, loader_i in enumerate(loaders):
            s_name = preparator.split_names[i]
            causal_nf_io.print_info(f"Testing {s_name} split")
            preparator.set_current_split(i)
            model.ckpt_name = ckpt_name
            _ = trainer.test(ckpt_path=ckpt_name, dataloaders=loader_i)
            metrics_stats = model.metrics_stats
            metrics_stats["current_epoch"] = trainer.current_epoch

            wandb_local.log_v2(
                {s_name: metrics_stats, "epoch": ckpt_name},
                root=trainer.logger.save_dir,
            )

    run.finish()
    if args.delete_ckpt:
        for f in glob.iglob(os.path.join(logger.save_dir, "*.ckpt")):
            causal_nf_io.print_warning(f"Deleting {f}")
            os.remove(f)

if args.pscf_analysis:
    ckpt_name_list = glob.glob(os.path.join(args.load_model, f"last.ckpt"))
    for ckpt_file in ckpt_name_list:
        model = causal_nf_train.load_model(
            cfg=cfg, preparator=preparator, ckpt_file=ckpt_file
        )
        model.eval()
        model.save_dir = dirpath
        ckpt_name = preparator.get_ckpt_name(ckpt_file)
        csv_col_names = []
        data_column_names = preparator.datasets[0].column_names
        max_col_vals = torch.max(preparator.datasets[0].x, dim=0).values
        min_col_vals = torch.min(preparator.datasets[0].x, dim=0).values
        for c in data_column_names:
            csv_col_names.append(f'{c}_F')
        for c in data_column_names:
            csv_col_names.append(f'{c}_CF')
        # Total treatment discrimination Y_TTD=Y(S^F, X^F, do(Z=Z^CF))
        csv_col_names.append(f'{data_column_names[-1]}_TTD')
        # Path-specific Y_PSDL=Y(do(S=S^CF), do(X=X^F, Z=Z^F))
        csv_col_names.append(f'{data_column_names[-1]}_PSDL')
        # Columns for Z^PSDT (path-specific treatment): Z^PSDT=Z(do(S=S^CF), do(X=X^F))
        for c in data_column_names[
                 preparator.num_sensitive + preparator.num_covariate:preparator.num_sensitive + preparator.num_covariate
                                                                     + preparator.num_treatment]:
            csv_col_names.append(f'{c}_PSDT')
        # Column for downstream effect of Z^PSDT => Y^PSDT = Y(S^F, X^F, do(Z=Z^PSDT))
        csv_col_names.append(f'{data_column_names[-1]}_PSDT')
        print(f"csv_col_names is of length {(len(csv_col_names))} are {csv_col_names}")
        # Variables for marginal risk
        samples_for_risk_marginal, samples_for_risk_marginal_fem, samples_for_risk_marginal_mal = None, None, None
        if (preparator.name in ['treatment_german', 'treatment_homecredit']) or 'synthetic' in preparator.name:
            csv_col_names.append('risk_score')
            csv_col_names.append('risk_score_treat_female')
            csv_col_names.append('risk_score_treat_male')
            K = min(500, preparator.datasets[0].x.shape[0])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(42)
            perm = torch.randperm(preparator.datasets[0].x.size(0), generator=generator)
            idx = perm[:K]
            samples_for_risk_marginal = torch.clone(preparator.datasets[0].x)[idx]
            # *********** #
            if preparator.name != 'treatment_german':
                dat_for_idx = preparator.datasets[0].x
            else:
                dat_for_idx = torch.cat([preparator.datasets[i].x for i in range(3)], dim=0)
            fem_dat = dat_for_idx[dat_for_idx[:, 0] < 1.0]
            mal_dat = dat_for_idx[dat_for_idx[:, 0] >= 1.0]
            K_f = min(500, fem_dat.shape[0])
            K_m = min(500, mal_dat.shape[0])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(42)
            perm_f = torch.randperm(fem_dat.size(0), generator=generator)
            perm_m = torch.randperm(mal_dat.size(0), generator=generator)
            idx_f, idx_m = perm_f[:K_f], perm_m[:K_m]
            samples_for_risk_marginal_fem = torch.clone(fem_dat)[idx_f]
            samples_for_risk_marginal_mal = torch.clone(mal_dat)[idx_m]
        for i, loader_i in enumerate(loaders):
            print(f"I am on {i} now")
            s_name = preparator.split_names[i]
            causal_nf_io.print_info(f"Testing {s_name} split")
            preparator.set_current_split(i)
            model.ckpt_name = ckpt_name
            to_save_data = []
            for j, (data, _) in enumerate(loader_i):
                row = []
                # Store the factual data (S^F, X^F, T^F, Y^F)
                factual_data = torch.clone(data)
                # Since we store this, floor it.
                factual_data[:, preparator.binary_dims] = torch.floor(factual_data[:, preparator.binary_dims])
                row.append(factual_data)  # Gets S^F, X^F, T^F, Y^F
                f_data = torch.clone(data)
                # Compute sensitive CF to get <S^CF,X^CF,T^CF,Y^CF>
                data_f = torch.clone(f_data)
                sens_cf_ = compute_sensitive_counterfactual(data_f, model, preparator)
                idxs = [x for x in range(preparator.num_sensitive, preparator.num_nodes)
                        if x in preparator.binary_dims]
                sens_cf_[:, idxs] = torch.clamp(sens_cf_[:, idxs], min=min_col_vals[idxs], max=max_col_vals[idxs])
                # Floor sensitive CF to save.
                sens_cf_data = torch.clone(sens_cf_)
                sens_cf_data[:, preparator.binary_dims] = torch.floor(sens_cf_data[:, preparator.binary_dims])
                row.append(sens_cf_data)  # Gets S^CF, X^CF, T^CF, Y^CF
                # Compute for total treatment discrimination Z^CF => Y(S^F, X^F, do(Z=Z^CF))
                data_f = torch.clone(f_data)
                path_treat_cf_ = compute_path_specific_downstream_effect_treatment_intervention(data_f, model,
                                                                                                preparator, sens_cf_)
                path_treat_cf_[:, -1] = torch.clamp(path_treat_cf_[:, -1], min=min_col_vals[-1], max=max_col_vals[-1])
                path_treat_cf_data = torch.clone(path_treat_cf_)
                path_treat_cf_data[:, preparator.binary_dims] = \
                    torch.floor(path_treat_cf_data[:, preparator.binary_dims])
                row.append(path_treat_cf_data[:, -1].view(-1, 1))  # Gets Y^TTD
                # Compute direct impact of S on Y using path-specific.
                data_f = torch.clone(f_data)
                path_direct_s_y_cf_ = compute_path_specific_sensitive_direct_label(data_f, model, preparator, sens_cf_)
                path_direct_s_y_cf_[:, -1] = torch.clamp(path_direct_s_y_cf_[:, -1],
                                                         min=min_col_vals[-1], max=max_col_vals[-1])
                path_direct_s_y_cf_data = torch.clone(path_direct_s_y_cf_)
                path_direct_s_y_cf_data[:, preparator.binary_dims] = \
                    torch.floor(path_direct_s_y_cf_data[:, preparator.binary_dims])
                row.append(path_direct_s_y_cf_data[:, -1].view(-1, 1))  # Gets Y^PSDL -- direct of S on Y
                # Computation of PSCF on Z => Z^PSDT(do(S=S^CF), do(X=X^F))
                data_f = torch.clone(f_data)
                pscf_z_ = compute_path_specific_treatment(data_f, model, preparator, sens_cf_)
                idxs = [x for x in range(preparator.num_sensitive + preparator.num_covariate,
                                         preparator.num_sensitive + preparator.num_covariate + preparator.num_treatment)
                        if x in preparator.binary_dims]
                pscf_z_[:, idxs] = torch.clamp(pscf_z_[:, idxs], min=min_col_vals[idxs],
                                               max=max_col_vals[idxs])
                pscf_z_data = torch.clone(pscf_z_)
                pscf_z_data[:, preparator.binary_dims] = torch.floor(pscf_z_data[:, preparator.binary_dims])
                # Save Z^PSDT
                row.append(pscf_z_data[:,
                           preparator.num_sensitive + preparator.num_covariate:
                           preparator.num_sensitive + preparator.num_covariate + preparator.num_treatment])
                # Compute downstream of Z^PSDT as Y(S^F, X^F, Z^PSDT)
                data_f = torch.clone(f_data)
                psdt_cf_ = compute_path_specific_downstream_effect_treatment_intervention(data_f, model, preparator,
                                                                                          pscf_z_)
                psdt_cf_[:, -1] = torch.clamp(psdt_cf_[:, -1], min=min_col_vals[-1], max=max_col_vals[-1])
                psdt_cf_data = torch.clone(psdt_cf_)
                psdt_cf_data[:, preparator.binary_dims] = \
                    torch.floor(psdt_cf_data[:, preparator.binary_dims])
                row.append(psdt_cf_data[:, -1].view(-1, 1))  # Gets Y^PSDT
                # Now we compute the marginalized risk score!
                cond = (preparator.name == 'treatment_german') or \
                       ('synthetic' in preparator.name) or \
                       (preparator.name == 'treatment_homecredit')
                if cond:
                    data_f = torch.clone(f_data)
                    risks = compute_marginal_risk_score(data_f, model, preparator, samples_for_risk_marginal)
                    row.append(risks.view(-1, 1))  # Gets risk_score
                    data_f = torch.clone(f_data)
                    risks = compute_marginal_risk_score(data_f, model, preparator, samples_for_risk_marginal_fem)
                    row.append(risks.view(-1, 1))  # Gets risk_score with female treatments
                    # Now we compute the marginalized risk score!
                    data_f = torch.clone(f_data)
                    risks = compute_marginal_risk_score(data_f, model, preparator, samples_for_risk_marginal_mal)
                    row.append(risks.view(-1, 1))  # Gets risk_score with male treatments
                to_save_data.append(torch.cat(row, dim=1))
            save_data = torch.cat(to_save_data, dim=0).numpy()
            df = pd.DataFrame(save_data, columns=csv_col_names)
            pscf_res_path = os.path.join(trainer.logger.save_dir, 'pscf_analysis')
            Path(pscf_res_path).mkdir(parents=True, exist_ok=True)
            df.to_csv(os.path.join(pscf_res_path, f'{s_name}.csv'), columns=csv_col_names, index=False)
print(f"Experiment folder: {logger.save_dir}")
