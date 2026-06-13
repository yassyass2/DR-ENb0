# Experiment Results

## Canonical Baseline vs SMOTE

Both runs use the same APTOS 2019 validation and held-out test splits, the same
green-channel preprocessing contract, and the same EfficientNetB0 architecture.
The baseline uses balanced class weights. The SMOTE run uses a class-balanced
synthetic training set and no class weights.

| Metric | Baseline | SMOTE | Delta |
| --- | ---: | ---: | ---: |
| QWK | 0.8655 | 0.8629 | -0.0026 |
| Accuracy | 0.7842 | 0.7869 | +0.0027 |
| Macro sensitivity | 0.6431 | 0.5997 | -0.0434 |
| Macro specificity | 0.9492 | 0.9483 | -0.0009 |
| Macro ROC-AUC | 0.9204 | 0.9083 | -0.0121 |

| Grade | Baseline sensitivity | SMOTE sensitivity | Delta |
| --- | ---: | ---: | ---: |
| 0 No DR | 0.9598 | 0.9648 | +0.0050 |
| 1 Mild | 0.5333 | 0.2667 | -0.2667 |
| 2 Moderate | 0.6207 | 0.7241 | +0.1034 |
| 3 Severe | 0.6471 | 0.5882 | -0.0588 |
| 4 Proliferative | 0.4545 | 0.4545 | +0.0000 |

## Interpretation

SMOTE does not improve the overall model for the current pipeline. Its QWK is
nearly tied with the baseline and its accuracy is slightly higher, but macro
sensitivity drops because grade 1 recall falls from 0.5333 to 0.2667. Grade 2
recall improves, while grade 4 recall is unchanged.

The selected report result should therefore remain the class-weighted baseline.
SMOTE should be reported as an imbalance-handling ablation that did not improve
minority-grade sensitivity overall.

## Explainability Comparison

The trained baseline model now has qualitative outputs for three XAI methods:

| Method | Artifact pattern | Notes |
| --- | --- | --- |
| Grad-CAM | `artifacts/figures/gradcam_*.png` | Original baseline XAI method; useful but sometimes attends to optic disc or border regions. |
| Grad-CAM++ | `artifacts/figures/gradcampp_*.png` | Uses stronger spatial weighting and often produces sharper heatmaps than Grad-CAM. |
| Score-CAM top-k | `artifacts/figures/scorecam_*.png` | Computed on a limited sample set using the top 32 activation channels per image to avoid 1280 masked forward passes per image. Outputs are broader and more diffuse. |

These artifacts support the proposal's qualitative XAI comparison. They do not
constitute lesion-mask validation: no IDRiD masks, pointing-game accuracy, or
IoU scores are used in the current scope.
