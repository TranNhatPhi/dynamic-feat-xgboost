"""
Optuna tuning — bộ hyperparameter CHUNG cho công bằng.

Chiến lược rẻ & chuẩn cho study: với mỗi dataset, Optuna tune hyperparameter XGBoost
(eta, max_depth, gamma, subsample, colsample) TRÊN plain-XGBoost / validation, rồi ÁP
CÙNG bộ đó cho MỌI cấu hình FE → cô lập đúng hiệu ứng của việc chọn FE.

Chạy:
  Mac:   ./.venv/bin/python -m src.tuning --trials 60 --jobs 8
  Cloud: python -m src.tuning --trials 100 --jobs 40   (dùng cả 40 nhân EPYC, CPU-only)
Xuất results/tuned_table1.csv (accuracy + F1-macro sau khi tune, mọi config).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import FeatXGBoost, FixedSelector, RoundRobinSelector
from src.preprocessing import prepare
from src.data_loader import UCIDataset
from src.select_datasets import hard_set

optuna.logging.set_verbosity(optuna.logging.WARNING)

CHEAP5 = ["identity", "random_proj", "ht_svd", "robust", "minmax"]
TWO = ["ht_svd", "random_proj"]
FOLDS = [0, 1, 2, 3]
SEED = 0
N_BOOST = 100


def tune_shared(sp, n_trials, n_jobs):
    """Tune hyperparameter XGBoost trên plain-XGBoost / validation của fold 0."""
    def objective(trial):
        params = {
            "eta": trial.suggest_float("eta", 1e-3, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        m = XGBClassifier(n_estimators=N_BOOST, verbosity=0, **params)
        m.fit(sp.X_train, sp.y_train)
        return accuracy_score(sp.y_val, m.predict(sp.X_val))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params


def build(cfg, tuned):
    if cfg == "plain_xgb":
        return None
    fe = {"fixed_htsvd": ["ht_svd"], "twofe": TWO, "cheap_rr": CHEAP5,
          "round_robin": FE_ARMS, "bandit": FE_ARMS}[cfg]
    if cfg == "fixed_htsvd":
        sel = FixedSelector(0)
    elif cfg == "bandit":
        sel = BanditSelector(FE_ARMS, alpha=0.25, seed=SEED)
    else:
        sel = RoundRobinSelector(len(fe))
    return FeatXGBoost(selector=sel, n_boost=N_BOOST, xgb_cfg=tuned,
                       lam=0.1, warmup=10, fe_names=fe, seed=SEED)


def eval_cfg(cfg, sp, tuned):
    if cfg == "plain_xgb":
        m = XGBClassifier(n_estimators=N_BOOST, verbosity=0, **tuned)
        m.fit(sp.X_train, sp.y_train)
        yp = m.predict(sp.X_test)
    else:
        m = build(cfg, tuned)
        m.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val, sp.X_test)
        yp = m.predict("test")
    return accuracy_score(sp.y_test, yp), f1_score(sp.y_test, yp, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    configs = ["plain_xgb", "fixed_htsvd", "twofe", "cheap_rr", "round_robin", "bandit"]
    rows = [["dataset", "config", "acc", "acc_std", "f1", "f1_std"]]
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    for name in hard_set():
        ds = UCIDataset(name)
        tuned = tune_shared(prepare(ds.get_split(0)), args.trials, args.jobs)
        for cfg in configs:
            accs, f1s = [], []
            for f in FOLDS:
                sp = prepare(ds.get_split(f))
                a, f1 = eval_cfg(cfg, sp, tuned)
                accs.append(a); f1s.append(f1)
            rows.append([name, cfg, round(np.mean(accs), 4), round(np.std(accs), 4),
                         round(np.mean(f1s), 4), round(np.std(f1s), 4)])
        print(f"{name}: tuned={tuned}", flush=True)
    with open(out / "tuned_table1.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("Đã ghi results/tuned_table1.csv")


if __name__ == "__main__":
    main()
