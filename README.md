# 🌌 MALLORN: Astrophysics-Informed TDE Classifier

> **MALLORN Challenge** - Giải pháp phân loại sự kiện TDE sử dụng Ensemble Learning kết hợp với các đặc trưng Vật lý Thiên văn chuyên sâu.

![Status](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Features](https://img.shields.io/badge/Physics-Informed-purple) ![License](https://img.shields.io/badge/License-MIT-green)

## 📑 Mục lục
1. [Giới thiệu](#-giới-thiệu)
2. [Điểm nổi bật (Key Features)](#-điểm-nổi-bật-key-features)
3. [Phương pháp Kỹ thuật](#-phương-pháp-kỹ-thuật)
4. [Cấu trúc Repository](#-cấu-trúc-repository)


---

## 🚀 Giới thiệu

Dự án này giải quyết bài toán phân loại **Tidal Disruption Events (TDE)** - hiện tượng hiếm gặp khi ngôi sao bị hố đen xé toạc. Khác với các phương pháp thuần dữ liệu (data-driven), giải pháp của chúng tôi tích hợp kiến thức **Vật lý thiên văn (Astrophysics)** để trích xuất các đặc trưng có ý nghĩa thực tế từ dữ liệu quang trắc (lightcurves) nhiễu và không đều.

Mục tiêu: Tối ưu hóa chỉ số **F1-Score** trên tập dữ liệu mất cân bằng nghiêm trọng (~5% TDE).

---

## ✨ Điểm nổi bật (Key Features)

Phiên bản nâng cấp (`improved-model-bonus-features`) mang đến những cải tiến vượt bậc:

* **🔭 Đặc trưng Vật lý Nâng cao:** Thay vì chỉ dùng độ sáng quan sát được, mô hình tính toán **Độ sáng tuyệt đối (Absolute Magnitude)** dựa trên Redshift ($z$) và Khoảng cách độ sáng (Luminosity Distance), giúp phân biệt năng lượng thực sự của vụ nổ.
* **🎨 Động học Màu sắc (Color Evolution):** Sử dụng Gaussian Process để mô hình hóa tốc độ làm nguội (**Cooling Rate**) của vật thể thông qua độ dốc màu ($g-r$) theo thời gian.
* **⚡ Tối ưu hóa GPU:** Hỗ trợ xử lý song song và huấn luyện XGBoost/CatBoost trên GPU để tăng tốc độ thử nghiệm.
* **⚖️ Xử lý Mất cân bằng:** Chiến lược **Cost-Sensitive Learning** với trọng số lớp động (Dynamic Class Weights) và Ngưỡng cắt thích ứng (Adaptive Thresholding).

---

## 🛠 Phương pháp Kỹ thuật

### 1. Feature Engineering (Trích xuất đặc trưng)
Quy trình xử lý dữ liệu chuyên sâu được thực hiện song song:

* **Mô hình hóa Bazin (Bazin Fitting):** Khớp đường cong ánh sáng vào hàm Bazin $F(t) = A \frac{e^{-(t-t_0)/\tau_{fall}}}{1 + e^{-(t-t_0)/\tau_{rise}}} + B$ để lấy tham số hình dạng vụ nổ ($t_{rise}, t_{fall}$).
* **Gaussian Process Regression (GP):** Nội suy dữ liệu bị khuyết để dự đoán chính xác Flux tại thời điểm cực đại (Peak) và 20 ngày sau đó.
* **Vật lý Vũ trụ:**
    * **Absolute Magnitude ($M_{abs}$):** Chuyển đổi Flux sang độ sáng tuyệt đối để loại bỏ ảnh hưởng của khoảng cách.
    * **Color Slope:** Tính tốc độ thay đổi màu sắc ($\Delta(g-r)/\Delta t$) để nhận diện đặc trưng làm nguội nhanh của TDE.
* **Thống kê:** Hệ số Stetson $J, K$ để đánh giá độ tin cậy của tín hiệu biến thiên.

### 2. Kiến trúc Mô hình (Ensemble Learning)
Sử dụng **Voting Classifier (Soft Voting)** kết hợp 3 mô hình Gradient Boosting mạnh nhất (SOTA):

| Mô hình | Vai trò & Cấu hình |
| :--- | :--- |
| **LightGBM** | Cơ chế **DART** (Dropouts) giúp chống Overfitting hiệu quả. |
| **XGBoost** | **Tree Method = 'hist'** (hỗ trợ GPU), tối ưu hóa tốc độ trên dữ liệu lớn. |
| **CatBoost** | Tự động xử lý đặc trưng phân loại và cân bằng dữ liệu (**SqrtBalanced**). |

### 3. Chiến lược Hậu xử lý (Post-processing)
* **Ngưỡng động (Dynamic Thresholding):** Thay vì ngưỡng cứng 0.5, ngưỡng quyết định được tính toán dựa trên phân vị (percentile) xác suất dự đoán, khớp với tỷ lệ TDE trong tập huấn luyện (~4.8%).

---

## 📂 Cấu trúc Repository

* **`improved-model-bonus-features.ipynb`**: **[RECOMMENDED]** Phiên bản cao cấp nhất chứa đầy đủ các đặc trưng vật lý và tối ưu hóa GPU.
* **`improved_model.ipynb`**: Phiên bản ổn định (Stable), tập trung vào các đặc trưng cơ bản và tối ưu hóa tham số Optuna.

---

