# Literature for CRPS-regularised rcGAN (MMGAN): agent index

**Project context.** MMGAN (rcGAN for weak-lensing mass-mapping) with the rcGAN ℓ1 + SD-reward regulariser
replaced or augmented by a CRPS loss over the P generated samples. Variants: standard, **fair (primary)**,
alpha-fair ("almost fair", AIFS-CRPS). Calibration: RCPS with the **Hoeffding** upper confidence bound.

**Layout.** `<group>/<NN_name>/src/` = LaTeX source (read this for the maths), `paper.pdf`, `ARXIV_ID`.
Line numbers below refer to the main `.tex` file. `fetch_papers.sh` re-downloads everything.

**Caveats**
- `08_WGAN-GP`: the authors uploaded **PDF only** (the `.tex` just wraps the PDF). Use `paper.pdf`; the WGAN-GP critic loss is also restated in LaTeX in rcGAN L401–460 and MMGAN L229–275.
- `11_Gneiting2007`: no source exists. `src/gneiting2007_transcribed.tex` is a **partial transcription by Claude** (Secs 2.1, 4.2, 4.3, 5.1, 5.2), with equation numbers matching the paper. Check against `paper.pdf`.
- `12_Ferro2014`: no source exists. `src/ferro2014_transcribed.tex` is a **full transcription by Claude**, with equation numbers matching the paper. Blocks marked [TRANSCRIBER NOTE] are additions, not the paper's text. Check against `paper.pdf`.
- `04` and `10`: the arXiv tarballs contain a duplicate nested copy of the source; use the top-level `src/main.tex`.

---

## CRPS / scoring-rule theory
| Folder | Paper | Where to look |
|---|---|---|
| suggested/11_ProperScoringRules_Gneiting2007 | Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation* (JASA 2007) | `src/gneiting2007_transcribed.tex`: propriety def eq (1); **CRPS** eqs (20)–(21), kernel form ½E\|X−X′\| − E\|X−x\|, strictly proper on finite-first-moment class; **energy score** (22)–(24); kernel scores Thm 4, eq (28); strict propriety Thm 5; Hoeffding-type/energy-distance inequalities (31)–(36). Positive orientation: loss = −CRPS. |
| suggested/12_FairScores_Ferro2014 | Ferro, *Fair scores for ensemble forecasts* (QJRMS 2014) | `src/ferro2014_transcribed.tex`: **fairness Def 1** (expected score optimised when ensemble dist. p = obs dist. q); ensemble-symmetric/finite Defs 2–3; **Thm 1** (fair binary scores, eq (1)) + proof in the Appendix (A1)–(A4); unfair Brier (2) vs **adjusted/fair Brier (3)**; threshold integral (5) → **fair CRPS**; TRANSCRIBER NOTE: kernel form fCRPS = (1/m)Σ\|x_i−y\| − 1/(2m(m−1))ΣΣ\|x_i−x_j\| and why the standard CRPS rewards under-dispersion; Fig. 2 (standard CRPS optimum α<β); dependent/exchangeable members Def 4, eq (6) (the adjusted score is fair for pairwise-uncorrelated members). |
| core/05_AIFS-CRPS_Lang2024 | Lang et al., *AIFS-CRPS* (ECMWF 2024) | `main.tex` L133–164: **CRPS eq (1)**, **fCRPS eq (2)** (1/(2M(M−1)) spread term), fCRPS degeneracy, **afCRPS_α eq (3)** = α·fCRPS + (1−α)·CRPS, ε=(1−α)/M, numerically stable positive-sum form eq (4); α=0.95 used (L169). |
| suggested/06_ScoringRuleGenNets_Pacchiardi2021 | Pacchiardi et al., *Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization* (JMLR 2024) | `paper.tex` L365–420: SR minimisation for conditional generators, kernel & energy scores, unbiased SGD; consistency theorems L483–672; **spatial SRs** (variogram, patched ES) L698–750; unbiased estimators App. L1942–2012; score defs App. L1889. |
| suggested/07_CramerGAN_Bellemare2017 | Bellemare et al., *The Cramér Distance as a Solution to Biased Wasserstein Gradients* (2017) | `distributional_l2.tex`: properties incl. unbiased sample gradients L175–220; **Wasserstein gradient bias** Thm L260; **Cramér distance** L287–335 (l₂² = ½ energy distance in 1-D, L325); Cramér GAN L380; proofs L481+. CRPS(F,y) = Cramér distance between F and δ_y. |

## Base model and our model
| Folder | Paper | Where to look |
|---|---|---|
| core/01_rcGAN_Bendel2022 | Bendel, Ahmad, Schniter, *A Regularized Conditional GAN for Posterior Sampling in Image Recovery Problems* (NeurIPS 2023) | `neurips_2023.tex`: W1 / cWGAN adversarial loss L401–460; **regulariser L469–512**: ℓ1 on the P-sample mean (eq LoneP) − β_SD × SD reward (eq LstdP); **Prop 3.1** (mean and covariance matching at β_SD = √(2/(πP(P+1)))) L500–512, proof App. L1252+; why not ℓ2 L570–670; β_SD auto-tuning L687–760. *Compare with CRPS: ℓ1 on the mean plus a pairwise-spread reward ≈ the E\|X−y\| − ½E\|X−X′\| structure.* |
| core/02_MMGAN_Whitney2024 | Whitney et al., *Generative modelling for mass-mapping with fast UQ* (2024) | `mass_mapping_rcGAN.tex`: lensing forward model/inverse problem L112–190; GAN/WGAN background L190–250; **rcGAN regulariser as used in MMGAN L277–336**; UQ from samples L356–370. |
| suggested/09_pcaGAN_Bendel2024 | Bendel et al., *pcaGAN* (NeurIPS 2024) | `arxiv.tex`: rcGAN objective restated L423–481; eigenvector/eigenvalue (covariance) regularisers L497–610. The closest competing regulariser. |
| suggested/08_WGAN-GP_Gulrajani2017 | Gulrajani et al., *Improved Training of Wasserstein GANs* (2017) | `paper.pdf` only: gradient-penalty critic objective (Sec 4, Alg 1). |

## Calibration: RCPS with Hoeffding
| Folder | Paper | Where to look |
|---|---|---|
| core/03_RCPS_Bates2021 | Bates et al., *Distribution-Free, Risk-Controlling Prediction Sets* (JACM 2021) | `main.tex`: RCPS definition L135; nested sets/monotone loss L172–195; UCB procedure and λ̂ L199–245; **simple Hoeffding bound L258–288 (Thm RCPS-from-Hoeffding)**; tighter Hoeffding/Bentkus/WSR L290–370 (alternatives); unbounded losses L391+. |
| core/04_Im2ImRCPS_Angelopoulos2022 | Angelopoulos et al., *Image-to-Image Regression with Distribution-Free UQ* (ICML 2022) | `main.tex`: pixel-wise nested intervals L177–200; RCPS guarantee L238–245; heuristic uncertainty (residual, Gaussian, quantile) L262–365; **Hoeffding UCB calibration and λ̂ L368–400**; algorithms in `algorithms/`. |
| suggested/10_ConformalIntro_Angelopoulos2021 | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* | Optional background: conformal risk control L831; general risk control / Learn-then-Test L1476–1540. |

## Not included (PDF-only, add if needed)
- Gneiting et al. (2008), *Assessing probabilistic forecasts of multivariate quantities*: energy score.
- Scheuerer & Hamill (2015): variogram score (dependence between pixels).
- Zamo & Naveau (2018), *Estimation of the CRPS for ensemble prediction systems*: estimator bias and variance.
