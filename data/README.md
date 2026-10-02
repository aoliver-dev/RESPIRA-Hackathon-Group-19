# ILA Progression Risk — Pulmonary Fibrosis Hackathon

Prototype for predicting progression from interstitial lung abnormalities (ILA) to clinically significant interstitial lung disease using baseline clinical variables plus a four-protein serum signature.

## Primary modelling cohort
- 650 patients with ILA
- 300 Progressive ILA
- 350 non-progressive ILA

Healthy controls and patients with established ILD are excluded from the primary prediction task.

## Predictors retained in the final model
### Baseline clinical variables
- age
- sex
- cigarette pack-years
- BMI
- baseline FVC
- baseline DLCO
- CT pattern

### Proteomic signature
- SFTPB
- GDF15
- MMP7
- SFTPD

Age, sex and BMI are retained intentionally for clinical completeness, even though they did not materially improve discrimination in earlier reduced-model comparisons.

## Variables explicitly excluded from prediction
Follow-up variables such as visit-2/visit-3 timing, follow-up FVC and delta FVC are not used because they would introduce post-baseline information leakage.

## Model
L2-regularized logistic regression with:
- median imputation for numeric variables
- standardization of numeric variables
- most-frequent imputation for categorical variables
- one-hot encoding of sex and CT pattern
- stratified 5-fold cross-validation

## Internal validation of expanded clinical + 4-protein model
- ROC-AUC: 0.9558
- Average precision: 0.9378
- Brier score: 0.0811
- Low-risk threshold: < 0.279
- High-risk threshold: >= 0.481

Observed progression rates:
- Low risk: 14/294 = 4.8%
- Intermediate risk: 23/59 = 39.0%
- High risk: 263/297 = 88.6%

These are internally cross-validated development results, not externally validated clinical cutoffs.

## Repository structure
- `src/` — modelling code
- `results/` — numerical outputs
- `figures/` — figures used during the hackathon
- `app/` — interactive prototype
- `presentation/` — current slide deck
- `docs/` — methodology and source notes

## Data availability
The participant-level TSV is intentionally not included. Add it only if the hackathon data-sharing terms explicitly allow public redistribution.
