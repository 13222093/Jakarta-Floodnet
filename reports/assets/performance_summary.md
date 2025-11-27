# Model Performance Summary

## 1. YOLOv8 (Flood Detection)
**Task:** Binary Classification (Flood / No Flood) based on CCTV images.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision** | **96.5%** | High reliability, very few false alarms. |
| **Recall** | **96.6%** | Misses almost no actual flood events. |
| **F1-Score** | **96.5%** | Balanced performance. |
| **mAP@50** | **0.972** | Mean Average Precision at 0.5 IoU. |

> **Verdict:** The vision model is highly reliable for real-time flood verification.

---

## 2. LSTM (Water Level Prediction)
**Task:** Time-series forecasting of water levels (cm) based on rainfall.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **MAE** | **0.68 cm** | Mean Absolute Error is less than 1 cm. |
| **RMSE** | **1.24 cm** | Root Mean Square Error indicates stability. |
| **Risk Accuracy** | **>99%** | Classification into AMAN/WASPADA/BAHAYA is nearly perfect. |

> **Verdict:** The physics-guided LSTM provides extremely precise water level forecasts, far exceeding the operational requirement of +/- 5cm accuracy.
