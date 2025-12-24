# 🌌 MALLORN: Tidal Disruption Event (TDE) Classifier

> **MALLORN Classifier Challenge** - Giải pháp sử dụng Ensemble Learning để phân loại sự kiện thiên văn TDE từ dữ liệu chuỗi thời gian ánh sáng (Lightcurves).

![Status](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📑 Mục lục
1. [Giới thiệu](#-giới-thiệu)
2. [Cấu trúc Repository](#-cấu-trúc-repository)
3. [Phương pháp Kỹ thuật](#-phương-pháp-kỹ-thuật)

---

## 🚀 Giới thiệu
Dự án này được xây dựng để giải quyết bài toán phân loại **Tidal Disruption Events (TDE)** trong dữ liệu thiên văn mô phỏng LSST. Thách thức chính bao gồm dữ liệu mất cân bằng (imbalanced data), chuỗi thời gian không đều (irregular sampling) và nhiễu từ bụi vũ trụ.

Mục tiêu là xây dựng mô hình Machine Learning có khả năng dự đoán chính xác sự kiện TDE (Target = 1) dựa trên các đặc trưng trích xuất từ Lightcurve đa bước sóng (u, g, r, i, z, y).

---

## 📂 Cấu trúc Repository
    
* **`improved_model.ipynb`**: **Main Pipeline (Production)**.
    * **Data Loading:** Tải và xử lý dữ liệu từ nhiều file rời rạc.
    * **Advanced Feature Engineering:** Trích xuất đặc trưng song song (Parallel Processing) sử dụng `joblib`. Bao gồm: tham số Bazin, hệ số Stetson, và dự đoán Gaussian Process.
    * **Hyperparameter Tuning:** Tự động tối ưu tham số cho LightGBM, XGBoost, CatBoost bằng **Optuna**.
    * **Ensemble Training:** Huấn luyện mô hình Voting Classifier kết hợp 3 model mạnh nhất.
    * **Submission:** Tạo file kết quả `submission_final.csv`.

---

## 🛠 Phương pháp Kỹ thuật

Giải pháp đạt hiệu năng cao nhờ sự kết hợp của các kỹ thuật tiên tiến:

### 1. Feature Engineering chuyên sâu cho Thiên văn
* **Bazin Fit:** Mô hình hóa hình dạng vụ nổ bằng hàm Bazin `Flux = A * (exp(-(t-t0)/tau_fall) / (1 + exp(-(t-t0)/tau_rise)))` để lấy thông tin về tốc độ tăng/giảm độ sáng đặc trưng của TDE.
* **Stetson Coefficients (J, K):** Đánh giá độ biến thiên tin cậy của tín hiệu, giúp phân biệt nhiễu và tín hiệu thực.
* **Gaussian Process Regression (GP):** Sử dụng GP kernel RBF để nội suy dữ liệu bị khuyết và dự đoán chính xác Flux tại thời điểm cực đại (Peak), từ đó tính toán màu sắc (Color indices) tin cậy.

### 2. Ensemble Learning mạnh mẽ
Sử dụng kiến trúc **Voting Classifier (Soft Voting)** kết hợp 3 mô hình Gradient Boosting hàng đầu:
* **LightGBM:** Tối ưu hóa tốc độ và hiệu năng trên dữ liệu lớn.
* **XGBoost:** Sử dụng `tree_method='hist'` và hỗ trợ GPU để tăng tốc độ huấn luyện.
* **CatBoost:** Xử lý tốt các đặc trưng phân loại và dữ liệu nhiễu.

### 3. Chiến lược tối ưu hóa
* **Optuna:** Tự động tìm kiếm siêu tham số (Hyperparameters) tối ưu cho từng mô hình thành phần thay vì chọn thủ công.
* **Imbalance Handling:** Sử dụng `scale_pos_weight` (căn chỉnh trọng số lớp dựa trên tỷ lệ mẫu) và `SqrtBalanced` để giúp mô hình học tốt lớp thiểu số (TDE).
* **Dynamic Threshold:** Áp dụng ngưỡng cắt động dựa trên phân vị xác suất dự đoán (percentile) thay vì ngưỡng cứng 0.5, giúp tối đa hóa Recall cho các sự kiện hiếm.

---
