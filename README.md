# MALLORN: Phân loại Sự kiện Gián đoạn Thủy triều (TDE)

> **MALLORN Classifier Challenge** - Giải pháp Ensemble Learning tối ưu hóa F1-Score cho bài toán phân loại thiên văn mất cân bằng dữ liệu.

![Status](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Ensemble](https://img.shields.io/badge/Model-LGBM%20%7C%20XGB%20%7C%20CatBoost-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Mục lục
1. [Tổng quan Dự án](#-tổng-quan-dự-án)
2. [Phương pháp Tiếp cận (Methodology)](#-phương-pháp-tiếp-cận-methodology)
3. [Hiệu suất & Kết quả](#-hiệu-suất--kết-quả)
4. [Cấu trúc Repository](#-cấu-trúc-repository)

---

## Tổng quan Dự án

### Bối cảnh & Thách thức
Kính thiên văn **LSST (Vera C. Rubin Observatory)** sắp đi vào hoạt động sẽ tạo ra kỷ nguyên dữ liệu lớn cho thiên văn học. Thách thức đặt ra là tự động phát hiện các sự kiện **Tidal Disruption Events (TDEs)** - hiện tượng hiếm gặp (chỉ chiếm ~5% dữ liệu) khi ngôi sao bị lỗ đen xé toạc, dựa trên dữ liệu ánh sáng (lightcurves) thưa và nhiễu.

### Mục tiêu
Xây dựng mô hình Machine Learning phân loại nhị phân (TDE vs Non-TDE) tối ưu hóa chỉ số **F1-Score**, đảm bảo cân bằng giữa khả năng phát hiện (Recall) và độ chính xác (Precision).

---

## Phương pháp Tiếp cận (Methodology)

Giải pháp của chúng tôi áp dụng kiến trúc **Ensemble Learning** kết hợp với **Feature Engineering chuyên sâu** trong lĩnh vực vật lý thiên văn.

### 1. Kỹ thuật Machine Learning (30%)
Chúng tôi sử dụng mô hình **Voting Classifier (Soft Voting)** kết hợp sức mạnh của 3 thuật toán Gradient Boosting hàng đầu:

* **LightGBM:** Tối ưu hóa tốc độ huấn luyện với cơ chế phát triển cây theo chiều lá (leaf-wise), phù hợp với dữ liệu dạng bảng lớn.
* **XGBoost:** Mạnh mẽ với khả năng Regularization (L1/L2) tốt, giảm thiểu overfitting trên dữ liệu nhiễu.
* **CatBoost:** Xử lý vượt trội các đặc trưng phân loại và tự động cân bằng dữ liệu (Auto Class Weights).

### 2. Cải tiến Mô hình & Xây dựng Đặc trưng (10%)
Thay vì sử dụng dữ liệu thô, chúng tôi trích xuất các đặc trưng nâng cao:
* **Mô hình hóa Bazin (Bazin Fitting):** Khớp đường cong ánh sáng vào hàm Bazin để trích xuất tham số hình dạng ($t_{rise}, t_{fall}$), giúp nhận diện đặc trưng "tăng nhanh, giảm chậm" của TDE.
* **Gaussian Process Regression (GP):** Nội suy dữ liệu bị khuyết để tính toán chính xác chỉ số màu ($g-r, u-g$) tại thời điểm cực đại.
* **Hệ số Stetson (J, K):** Phân biệt biến thiên tín hiệu thực với nhiễu ngẫu nhiên.
* **Tối ưu hóa Hyperparameter:** Sử dụng **Optuna** để tự động tìm kiếm bộ tham số tối ưu nhất cho từng mô hình thành phần.

### 3. Chiến lược Xử lý Mất cân bằng
* Áp dụng **Class Weights** (`scale_pos_weight`, `SqrtBalanced`) để tăng trọng số cho lớp thiểu số TDE.
* Sử dụng **Dynamic Thresholding** (Ngưỡng động): Ngưỡng quyết định được chọn dựa trên phân vị xác suất (percentile) thay vì ngưỡng cứng 0.5, giúp tối đa hóa Recall.

---

## Hiệu suất & Kết quả (20%)

Mô hình được đánh giá thông qua chiến lược **Stratified K-Fold Cross-Validation (5 Folds)** để đảm bảo độ tin cậy.

| Metric | Giá trị | Nhận xét |
| :--- | :--- | :--- |
| **CV F1-Score** | **~0.6400** | Cải thiện đáng kể so với mô hình đơn lẻ (~0.62). |
| **Precision** | Ổn định | Giảm thiểu báo động giả (False Positives). |
| **Recall** | Cao | Bắt được tối đa các sự kiện TDE tiềm năng. |

**Phân phối dự đoán:** Trên tập Test, mô hình dự đoán **5.35%** số lượng vật thể là TDE, tương đồng cao với tỷ lệ thực tế trong tập Train (**4.86%**), chứng tỏ mô hình không bị thiên kiến (bias).

---

## Cấu trúc Repository

* `mallorn.ipynb`: **EDA & Baseline**. Phân tích thống kê, trực quan hóa dữ liệu và kiểm thử ý tưởng ban đầu.
* `improved_model.ipynb`: **Production Pipeline**. Chứa toàn bộ quy trình từ xử lý dữ liệu, trích xuất đặc trưng song song, tối ưu tham số Optuna đến huấn luyện Ensemble và xuất kết quả.
