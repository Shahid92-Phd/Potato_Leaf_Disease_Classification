"""
S04 - Statistical analysis replacing metric-as-block testing  (R2.3, R2.6)

A. Sample-level paired comparison on the SAME 300 test images:
   exact McNemar test (MRV-Net vs each available baseline), Holm-adjusted,
   with paired-bootstrap 95% CIs of the accuracy and macro-F1 differences.
B. 95% CIs for every model: Wilson (accuracy) and percentile bootstrap (accuracy, macro-F1).
C. Independent-run comparison of the ablation variants (seeds as blocks): mean +/- SD,
   Friedman test, and Holm-adjusted paired t-tests / Wilcoxon tests of each variant vs full M+R+V.
D. (Optional) Reproduction of the original Friedman aligned-ranks analysis from a metrics table,
   documenting that n blocks were used: put a CSV 'original_metrics.csv' with rows = models and
   columns = the metrics that were entered into STAC.
All analyses are repeated on the de-duplicated test subset (results/test_dedup.csv) if it differs.
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from common import (ensure_dirs, load_split, load_subset, load_model, available_baselines, predict_probs,
                    save_preds, metrics_from_preds, wilson_ci, save_json)

B = 2000


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def mcnemar_exact(a_correct, b_correct):
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
    return b, c, p


def macro_f1(y, yhat):
    from sklearn.metrics import f1_score
    return f1_score(y, yhat, average="macro", labels=[0, 1, 2], zero_division=0)


def boot_ci(y, preds, rng, fn):
    n = len(y)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(fn(y[i], *[p[i] for p in preds]))
    return np.percentile(vals, [2.5, 97.5])


def get_test_predictions():
    split = load_split()
    preds = {}
    f = os.path.join(C.PRED_DIR, "MRV-Net_saved.csv")
    x = None
    if not os.path.exists(f):
        x, y, d = load_subset(split, "test")
        save_preds("MRV-Net_saved", d, predict_probs(load_model(C.MRVNET_PATH), x, C.MRVNET_INPUT_MODE))
    preds["MRV-Net"] = pd.read_csv(f)
    for name, path in available_baselines().items():
        f = os.path.join(C.PRED_DIR, f"{name}_saved.csv")
        if not os.path.exists(f):
            size = C.BASELINE_IMG_SIZE.get(name, C.IMG_SIZE)
            xb, y, d = load_subset(split, "test", size)
            save_preds(f"{name}_saved", d, predict_probs(load_model(path), xb, C.BASELINE_INPUT_MODE[name]))
        preds[name] = pd.read_csv(f)
    return preds


def section_ab(preds, subset_paths=None, tag="full"):
    rng = np.random.default_rng(C.RANDOM_STATE)
    ref = preds["MRV-Net"]
    keep = np.ones(len(ref), bool) if subset_paths is None else ref.path.isin(subset_paths).values
    y = ref.label.values[keep]
    pm = ref.pred.values[keep]
    per_model, comps = [], []
    for name, d in preds.items():
        d = d.set_index("path").loc[ref.path].reset_index()
        assert (d.label.values == ref.label.values).all(), f"label/order mismatch for {name}"
        p = d.pred.values[keep]
        k = int((p == y).sum())
        lo, hi = wilson_ci(k, len(y))
        bl, bh = boot_ci(y, [p], rng, lambda yy, pp: (yy == pp).mean())
        fl, fh = boot_ci(y, [p], rng, macro_f1)
        per_model.append(dict(model=name, n=len(y), accuracy=k / len(y), wilson_lo=lo, wilson_hi=hi,
                              boot_acc_lo=bl, boot_acc_hi=bh, macro_f1=macro_f1(y, p),
                              boot_f1_lo=fl, boot_f1_hi=fh))
        if name == "MRV-Net":
            continue
        b, c, pv = mcnemar_exact(pm == y, p == y)
        dl, dh = boot_ci(y, [pm, p], rng, lambda yy, a, bb: (a == yy).mean() - (bb == yy).mean())
        fdl, fdh = boot_ci(y, [pm, p], rng, lambda yy, a, bb: macro_f1(yy, a) - macro_f1(yy, bb))
        comps.append(dict(comparison=f"MRV-Net vs {name}", mrv_only_correct=b, other_only_correct=c,
                          discordant=b + c, p_exact=pv,
                          acc_diff=(pm == y).mean() - (p == y).mean(), acc_diff_lo=dl, acc_diff_hi=dh,
                          f1_diff=macro_f1(y, pm) - macro_f1(y, p), f1_diff_lo=fdl, f1_diff_hi=fdh))
    pm_df = pd.DataFrame(per_model).sort_values("accuracy", ascending=False)
    cp = pd.DataFrame(comps)
    if len(cp):
        cp["p_holm"] = holm(cp.p_exact)
        cp["significant_0.05"] = np.where(cp.p_holm < 0.05, "Yes", "No")
        cp = cp.sort_values("p_exact")
    pm_df.to_csv(os.path.join(C.RESULTS_DIR, f"stats_model_CIs_{tag}.csv"), index=False)
    cp.to_csv(os.path.join(C.RESULTS_DIR, f"stats_mcnemar_{tag}.csv"), index=False)
    print(f"\n[{tag}] per-model CIs\n", pm_df.round(4).to_string(index=False))
    if len(cp):
        print(f"\n[{tag}] McNemar (Holm)\n", cp.round(4).to_string(index=False))


def section_c():
    f = os.path.join(C.RESULTS_DIR, "ablation_runs.csv")
    if not os.path.exists(f):
        print("No ablation_runs.csv; skipping section C")
        return
    df = pd.read_csv(f)
    ref = "M+R+V"
    out = []
    for metric in ("test_accuracy", "test_macro_f1"):
        piv = df.pivot_table(index="seed", columns="config", values=metric)
        if ref not in piv:
            continue
        rows = []
        for cfg in piv.columns:
            if cfg == ref:
                continue
            pair = piv[[ref, cfg]].dropna()
            if len(pair) < 2:
                continue
            dlt = pair[ref] - pair[cfg]
            t = stats.ttest_rel(pair[ref], pair[cfg]) if dlt.std() > 0 else None
            try:
                w = stats.wilcoxon(pair[ref], pair[cfg]).pvalue if (dlt != 0).any() else 1.0
            except ValueError:
                w = 1.0
            se = dlt.std(ddof=1) / np.sqrt(len(dlt))
            tc = stats.t.ppf(0.975, len(dlt) - 1)
            rows.append(dict(metric=metric, comparison=f"{ref} vs {cfg}", n_seeds=len(dlt),
                             mean_ref=pair[ref].mean(), mean_other=pair[cfg].mean(),
                             mean_diff=dlt.mean(), ci_lo=dlt.mean() - tc * se, ci_hi=dlt.mean() + tc * se,
                             p_ttest=(t.pvalue if t is not None else 1.0), p_wilcoxon=w))
        r = pd.DataFrame(rows)
        if len(r):
            r["p_ttest_holm"] = holm(r.p_ttest)
            r["p_wilcoxon_holm"] = holm(r.p_wilcoxon)
            out.append(r)
        branch = [c for c in ["M", "R", "V", "M+R", "M+V", "R+V", "M+R+V"] if c in piv]
        full = piv[branch].dropna()
        if len(branch) >= 3 and len(full) >= 2:
            fr = stats.friedmanchisquare(*[full[c] for c in branch])
            out.append(pd.DataFrame([dict(metric=metric, comparison="Friedman over branch configs (seeds=blocks)",
                                          n_seeds=len(full), p_ttest=fr.pvalue, mean_diff=fr.statistic)]))
    if out:
        res = pd.concat(out)
        res.to_csv(os.path.join(C.RESULTS_DIR, "stats_ablation_seeds.csv"), index=False)
        print("\nAblation (independent runs)\n", res.round(4).to_string(index=False))


def friedman_aligned(M):
    """M: n_blocks x k_models (higher = better). Returns statistic, p, mean ranks, z-SE (Garcia et al. 2010)."""
    M = np.asarray(M, float)
    n, k = M.shape
    A = M - M.mean(1, keepdims=True)
    R = stats.rankdata(A.ravel()).reshape(n, k)       # larger aligned value -> larger rank
    Rj, Ri = R.sum(0), R.sum(1)
    num = (k - 1) * ((Rj ** 2).sum() - (k * n ** 2 / 4) * (k * n + 1) ** 2)
    den = (k * n * (k * n + 1) * (2 * k * n + 1) / 6) - (Ri ** 2).sum() / k
    T = num / den
    return T, stats.chi2.sf(T, k - 1), Rj / n, np.sqrt(k * (n * k + 1) / 6)


def section_d():
    f = "original_metrics.csv"
    if not os.path.exists(f):
        return
    t = pd.read_csv(f, index_col=0)          # rows models, cols metrics
    T, p, mr, se = friedman_aligned(t.T.values)
    res = pd.DataFrame({"model": t.index, "mean_aligned_rank": mr}).sort_values("mean_aligned_rank", ascending=False)
    top = res.iloc[0]
    res["z_vs_top"] = (top.mean_aligned_rank - res.mean_aligned_rank) / se
    res["p_unadj"] = 2 * stats.norm.sf(res.z_vs_top.abs())
    res.loc[res.index != top.name, "p_holm"] = holm(res.loc[res.index != top.name, "p_unadj"])
    res.to_csv(os.path.join(C.RESULTS_DIR, "stats_friedman_aligned_reproduction.csv"), index=False)
    save_json(dict(n_blocks=t.shape[1], k_models=t.shape[0], statistic=T, p=p,
                   mean_of_mean_ranks=float(mr.mean()), expected_mean=(t.shape[1] * t.shape[0] + 1) / 2),
              os.path.join(C.RESULTS_DIR, "stats_friedman_aligned_reproduction.json"))
    print(f"\nFriedman aligned ranks: n={t.shape[1]} blocks, k={t.shape[0]}, T={T:.5f}, p={p:.5f}")
    print(res.round(5).to_string(index=False))


def main():
    ensure_dirs()
    preds = get_test_predictions()
    section_ab(preds, None, "full")
    dd = os.path.join(C.RESULTS_DIR, "test_dedup.csv")
    if os.path.exists(dd):
        paths = pd.read_csv(dd).path.values
        if 0 < len(paths) < len(preds["MRV-Net"]):
            section_ab(preds, paths, "dedup")
    section_c()
    section_d()


if __name__ == "__main__":
    main()
