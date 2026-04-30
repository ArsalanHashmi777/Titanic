# MNIST Digit Recognition: Classical ML Baseline

This project establishes a high-performance baseline for digit classification using Classical Machine Learning techniques. By comparing **Random Forest** and **XGBoost**, I achieved a 97% accuracy rate while identifying key spatial bottlenecks that set the stage for Deep Learning.

## 🚀 Performance Summary
| Model | Accuracy | Environment | Key Insight |
| :--- | :--- | :--- | :--- |
| Random Forest | 96.74% | Local CPU | Robust baseline, high confusion on 4 vs 9. |
| **XGBoost** | **97.00%** | **Kaggle GPU** | **Champion Model**; utilized GPU acceleration. |

## 🔍 Feature Importance Visualization
To understand how the models "see" digits, I generated heatmaps of pixel importance.

### Random Forest Importance
![Random Forest Heatmap](path/to/your/rf_heatmap.png)
*Observation: Shows a broad, slightly blurred focus on the central drawing area.*

### XGBoost Importance (Champion)
![XGBoost Heatmap](path/to/your/xgb_heatmap.png)
*Observation: Much sharper focus on specific "stroke" pixels, explaining the higher accuracy.*

## 🛠️ Modular Pipeline
This project utilizes a custom `data_preprocessing.py` utility script to handle:
* Scaled normalization (0-1 range).
* Label encoding for XGBoost compatibility.
* Automated train/test splitting.

## 🏁 Final Checklist
- [x] **Baseline:** Random Forest (96.74%).
- [x] **Champion:** XGBoost (97.00%).
- [x] **Error Analysis:** Identified spatial confusion hotspots (4 vs. 9).
- [x] **Next Steps:** Moving to CNNs for 99%+ accuracy.
