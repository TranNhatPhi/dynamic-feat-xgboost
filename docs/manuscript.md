# Rethinking Embedded Feature Engineering in Gradient Boosting: A Static Cheap Rotation Rivals Ensemble and Adaptive Selection

*(Working draft — Empirical Study. Numbers are from fixed-hyperparameter experiments on 15 high-dimensional UCI datasets, 4-fold CV × 3 seeds. **All citations marked `[CITE]` must be verified before submission.**)*

---

## Abstract

Embedding feature engineering (FE) into the boosting loop—rather than applying it once as preprocessing—has been shown to improve gradient-boosted trees. A recent method, Feat-XGBoost `[CITE]`, rotates an *ensemble* of six FE transforms across boosting iterations. This design raises two unexamined questions: is the full six-transform ensemble necessary, and would *adaptively* selecting a transform per iteration (rather than blindly rotating) do better? We conduct a controlled empirical study on 15 high-dimensional UCI classification datasets (4-fold cross-validation, 3 seeds), comparing six selection strategies within a single boosting framework: no FE, one fixed transform, a 2-transform rotation, a cheap 5-transform rotation, the original 6-transform round-robin, and a contextual-bandit selector. We find that (i) FE diversity does help—round-robin significantly outperforms plain XGBoost (Δacc = +2.4 pts, paired *t* = 2.57) and single-transform variants; (ii) the most expensive transform (Autofeat) is nearly free of benefit—dropping it reduces peak feature-representation memory by **57 %** with no significant accuracy change (Δacc = −0.16 pts, *t* = −0.56); and (iii) online adaptive selection provides **no advantage**: a contextual bandit ties the cheap static rotation on accuracy (*t* = −0.37) while consuming as much memory as the full ensemble, because it must materialize every candidate transform. An oracle analysis explains the negative result: the per-iteration optimal transform is nearly uniform across boosting phases, so there is little exploitable structure. We recommend a simple static rotation over cheap transforms and release all code and data splits.

**Keywords:** gradient boosting, feature engineering, XGBoost, contextual bandit, empirical study, computational efficiency.

---

## 1. Introduction

Feature engineering (FE) frequently determines the effectiveness of tabular machine-learning models: even strong classifiers struggle when the input representation obscures class structure `[CITE]`. In the gradient-boosting family (XGBoost `[CITE: Chen & Guestrin, 2016]`, LightGBM `[CITE: Ke et al., 2017]`), FE has traditionally been a *preprocessing* step applied once, decoupled from the learner. A recent line of work instead embeds FE *inside* the boosting loop: at each iteration a feature transform is applied to (a subsample of) the training data before the next tree is fit. Feat-XGBoost `[CITE]` implements this idea, rotating six transforms—identity, Autofeat, random projection, a hard-thresholded SVD, robust scaling, and min–max scaling—across iterations, and reports accuracy gains over standard baselines on UCI benchmarks.

While effective, this design leaves two questions unanswered:

1. **Is the full ensemble necessary?** Rotating six transforms—some of them expensive (Autofeat expands the feature space several-fold)—incurs substantial computational cost. It is unclear how much each transform contributes.
2. **Is blind rotation optimal?** The transform is chosen by a fixed schedule (iteration index modulo six), ignoring the state of the data. A natural hypothesis is that *adaptively* choosing the transform per iteration—e.g., denoising early and preserving raw features later—would be superior.

We answer these questions with a controlled empirical study rather than by proposing a new state-of-the-art method. Our study is organized around four research questions:

- **RQ1.** Does FE diversity in the boosting loop help, relative to no FE or a single transform?
- **RQ2.** Is the full six-transform ensemble needed, or does a cheaper subset suffice?
- **RQ3.** Does adaptive per-iteration selection (a contextual bandit) outperform fixed selection?
- **RQ4.** Does the optimal transform vary systematically across boosting iterations, and can this be exploited online?

**Contributions.** (1) A fair, single-framework comparison of six FE-selection strategies over 15 high-dimensional datasets with statistical testing. (2) Evidence that the most expensive transform (Autofeat) can be removed with a 57 % reduction in feature-representation memory and no significant accuracy loss. (3) A negative but well-supported result: online adaptive selection does not beat a cheap static rotation, with an oracle analysis explaining *why*. (4) A concrete practical recommendation and an open, reproducible codebase.

## 2. Related Work

**Feature engineering for boosting.** `[CITE]` … (automatic feature construction, e.g., Autofeat `[CITE: Horn et al., 2020]`; dimensionality reduction such as random projection `[CITE]` and SVD-based denoising `[CITE]`). Prior work largely treats FE as preprocessing; Feat-XGBoost `[CITE]` is the closest to ours in embedding FE within boosting.

**Gradient-based one-side sampling (GOSS).** Introduced in LightGBM `[CITE: Ke et al., 2017]`, GOSS keeps large-gradient (hard) samples and subsamples small-gradient ones with re-weighting; Feat-XGBoost and our framework use it to focus each iteration on hard examples.

**Contextual bandits.** LinUCB `[CITE: Li et al., 2010]` and related algorithms select actions from context to maximize cumulative reward. We repurpose a contextual bandit to select an FE transform per boosting iteration; to our knowledge this specific use has not been systematically evaluated.

## 3. Materials and Methods

### 3.1 Unified framework and selection strategies

All strategies share one boosting engine that differs only in an interchangeable *FE selector*. At iteration $t$: (1) compute gradients of the multiclass log-loss on the training set; (2) apply GOSS ($a=0.4$ large-gradient, $b=0.3$ small-gradient, re-weighting factor $(1-a)/b$) to obtain a hard subset; (3) a selector picks one transform $\phi_t$ from a candidate set; (4) a single boosting round (`multi:softprob`, one round) is fit on the transformed data and added to the additive margin. At inference, each stored tree is applied in its own transformed space. Candidate transforms follow Feat-XGBoost: identity, Autofeat (first-round only: $1/x, x^2, x^3, e^x$), Gaussian random projection, hard-thresholded SVD (Gavish–Donoho threshold `[CITE]`), robust scaling, min–max scaling.

We compare six selectors:

| Strategy | Candidate transforms | Selection |
|---|---|---|
| Plain XGBoost | — | no FE |
| Fixed-HT-SVD | 1 (HT-SVD) | constant |
| 2-FE rotation | 2 (HT-SVD, random proj.) | round-robin |
| Cheap rotation | 5 (all but Autofeat) | round-robin |
| Round-robin (Feat-XGBoost) | 6 | round-robin ($t \bmod 6$) |
| Contextual bandit | 6 | LinUCB on hard-subset context |

The bandit's context is a low-dimensional summary of the hard subset (mean/std gradient magnitude, class coverage/entropy, dimensionality); its reward is validation-accuracy improvement minus a cost penalty $\lambda \cdot \text{cost}(\phi)$. A 10-iteration warm-up forces round-robin so the bandit observes every arm.

### 3.2 Data and protocol

We use 15 high-dimensional datasets from the standardized UCI benchmark `[CITE: Fernández-Delgado et al., 2014]` (splits from `[CITE]`), excluding one dataset with a train/test label mismatch. Each dataset is evaluated with 4-fold cross-validation × 3 seeds; we report mean ± std accuracy (and macro-F1). Because our claims are *relative* (strategy vs. strategy under identical conditions), we fix boosting hyperparameters ($\eta=0.3$, max depth $=6$, 100 iterations) across all strategies; we discuss tuning in Limitations.

### 3.3 Cost measurement

We report two physical costs measured on a single machine (Apple M4, 10 cores): wall-clock fit time (mean of 3 repeats) and the memory footprint of the precomputed feature representations. Crucially, each strategy precomputes **only the transforms it can use**; the bandit, however, must precompute all six (its action menu), which is reflected in its measured cost.

## 4. Results

### 4.1 RQ1–RQ2: diversity helps, but the full ensemble is wasteful

Table 1 reports mean accuracy and cost across strategies (increasing FE diversity). Accuracy rises with diversity from plain XGBoost (0.757) to round-robin (0.781). Round-robin significantly beats plain XGBoost (Δ = +0.024, paired *t* = 2.57, 12/15 wins) and the single-transform Fixed-HT-SVD (11/15 wins), confirming **RQ1**: FE diversity in the boosting loop is beneficial.

However, the gain saturates. Dropping Autofeat—the most expensive transform—moves from round-robin (6 transforms) to the cheap 5-transform rotation with **no significant accuracy change** (Δ = −0.0016, *t* = −0.56, i.e. statistically indistinguishable) while reducing precomputed-representation memory from 5.06 MB to 2.19 MB, a **57 % reduction**, and reducing fit time by ~15 %. This answers **RQ2**: the full ensemble is unnecessary; Autofeat's cost is not justified by its accuracy contribution.

**Table 1.** Mean accuracy and cost over 15 datasets (4-fold × 3 seeds; fixed hyperparameters).

| Strategy | #FE | Accuracy | Fit time (s) | FE memory (MB) |
|---|---:|---:|---:|---:|
| Plain XGBoost | 0 | 0.757 | 1.04 | 1.02 |
| Fixed-HT-SVD | 1 | 0.767 | 1.30 | 0.19 |
| 2-FE rotation | 2 | 0.771 | 1.27 | 0.47 |
| **Cheap rotation** | 5 | **0.779** | 1.25 | **2.19** |
| Round-robin (orig.) | 6 | 0.781 | 1.47 | 5.06 |
| Contextual bandit | 6* | 0.778 | 1.33 | 5.06 |

*\*The bandit chooses one transform per iteration but must materialize all six.*

### 4.2 RQ3: adaptive selection does not help—and is dominated

The contextual bandit matches the cheap static rotation on accuracy (Δ = −0.0012, *t* = −0.37; 9/6 split, no significant difference). It exceeds the impoverished 2-FE (*t* = 2.16) and single-FE (*t* = 2.04) baselines, but only because those are too narrow; it never beats the cheap rotation. Moreover, on the accuracy–cost plane (Fig. 1) the bandit is **Pareto-dominated** by the cheap rotation: it uses the same memory as the full round-robin (5.06 MB, since it must precompute Autofeat as a candidate) yet is *less* accurate than the cheaper rotation. This answers **RQ3**: online adaptive selection provides no benefit here and is strictly worse in cost than a fixed cheap rotation.

**Figure 1.** Accuracy vs. FE memory (Pareto frontier). The cheap rotation is the knee; the bandit lies below the frontier. *(see `results/fig1_pareto.png`)*

### 4.3 RQ4: why adaptivity fails

To test whether *any* per-iteration selector could help, we compute a greedy oracle that, at each iteration, tries all six transforms and keeps the one minimizing validation loss. Aggregated over datasets, the oracle's transform distribution is **nearly uniform across boosting phases** (each transform 13–36 %), with only a mild monotone trend: Autofeat is modestly favored early (22 %) and declines late (13 %), while identity is common throughout (30–36 %). The clean "denoise-early, preserve-late" pattern holds only for individual noisy datasets (e.g., hill-valley), not in aggregate. With no strong temporal structure and a noisy per-iteration reward signal, an online learner has little to exploit—explaining the RQ3 result.

## 5. Discussion

### 5.1 Practical recommendation

For embedding FE into gradient boosting, practitioners should use a **static rotation over cheap transforms** (dimensionality reducers and scalers), omitting expensive feature-expansion methods such as Autofeat. This retains the accuracy benefit of FE diversity at roughly one-third the memory of the full ensemble and requires no learned controller.

### 5.2 Why the bandit was worth testing—and why it lost

Blind rotation is an obvious target for improvement, and a cost-aware bandit *can* concentrate on cheap transforms. But two factors neutralize it: (a) it must precompute every candidate transform to keep them selectable, erasing its cost advantage; and (b) the per-iteration signal is too noisy and too weakly structured (RQ4) for the learned policy to beat a fixed schedule on accuracy. This is a useful negative result: it delimits when adaptive FE selection is *not* worth its complexity.

### 5.3 Limitations

Our comparison is *relative* under fixed hyperparameters; absolute numbers would shift with per-configuration tuning (e.g., Optuna), which we did not run and leave as future work. We study 15 high-dimensional datasets and XGBoost only; scaling, base-learner generality (LightGBM, random forests), and larger datasets remain open. Timing was measured on small datasets where fit time is dominated by fixed overheads, so the runtime gap is modest; the memory gap is the more robust cost signal.

## 6. Conclusion

Embedding feature engineering into gradient boosting helps, but the popular six-transform ensemble is over-engineered: a cheap static rotation matches it at a fraction of the cost, and a learned online selector adds complexity without benefit. We recommend the simple cheap rotation and release code, data splits, and all experimental artifacts for reproducibility.

---

## References (TO VERIFY — do not submit unchecked)

- Chen, T., Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD 2016. `[verify]`
- Ke, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017. `[verify]`
- Li, L. et al. A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010. `[verify]`
- Fernández-Delgado, M. et al. Do we need hundreds of classifiers to solve real world classification problems? JMLR 2014. `[verify]`
- Horn, F., Pack, R., Rieger, M. The autofeat Python Library for Automated Feature Engineering and Selection. 2020. `[verify]`
- Gavish, M., Donoho, D. The Optimal Hard Threshold for Singular Values is 4/√3. IEEE TIT 2014. `[verify]`
- Feat-XGBoost (base paper). Pattern Recognition, 2026. DOI:10.1016/j.patcog.2026.113169. `[verify authors/title]`

*Note: All bibliographic details above are provided as leads and MUST be checked against the actual sources before submission (per academic integrity — do not cite unverified).*
