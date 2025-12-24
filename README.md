# 🌌 MALLORN: Astrophysics-Informed TDE Classifier

> **MALLORN Classifier Challenge** - Giải pháp phân loại sự kiện TDE (Tidal Disruption Events) sử dụng Ensemble Learning kết hợp với các đặc trưng Vật lý Thiên văn chuyên sâu.

![Status](https://img.shields.io/badge/Status-Completed-success) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Type](https://img.shields.io/badge/Type-Physics_Informed_ML-purple) ![Device](https://img.shields.io/badge/Device-CPU%20%26%20GPU-orange)

## 📑 Mục lục
1. [Giới thiệu](#-giới-thiệu)
2. [Điểm nổi bật (Key Innovations)](#-điểm-nổi-bật-key-innovations)
3. [Phương pháp Kỹ thuật](#-phương-pháp-kỹ-thuật)
4. [Hiệu suất Mô hình](#-hiệu-suất-mô-hình)
5. [Cấu trúc Repository](#-cấu-trúc-repository)


---

## 🚀 Giới thiệu

**Tidal Disruption Events (TDEs)** là hiện tượng thiên văn hiếm gặp khi một ngôi sao bị lực thủy triều của hố đen siêu khối lượng xé toạc. Thách thức của bài toán MALLORN là tự động phát hiện TDE từ dữ liệu quang trắc (lightcurves) với các đặc điểm khó:
* **Dữ liệu cực kỳ mất cân bằng:** TDE chỉ chiếm ~4.8% tập dữ liệu.
* **Dữ liệu thưa và nhiễu:** Chuỗi thời gian không đều, nhiều khoảng trống.
* **Metadata yếu:** Các thông tin như Redshift nếu dùng thô sơ sẽ không phân tách được các lớp dữ liệu.

**Mục tiêu:** Xây dựng mô hình phân loại nhị phân tối ưu hóa chỉ số **F1-Score**, chuyển đổi từ phương pháp thuần dữ liệu sang hướng tiếp cận **định hướng vật lý (Physics-Informed)**.

---

## ✨ Điểm nổi bật (Key Innovations)

Phiên bản nâng cấp (`improved-model-bonus-features`) mang đến những cải tiến mang tính chiến lược:

* **🔭 Đặc trưng Vật lý Nâng cao:** Thay vì chỉ sử dụng độ sáng quan sát (Flux), chúng tôi kết hợp với **Redshift ($z$)** để tính toán **Độ sáng tuyệt đối ($M_{abs}$)**. Điều này giúp mô hình phân biệt được năng lượng thực sự của một vụ nổ lớn ở xa so với một biến quang nhỏ ở gần.
* **🌡️ Động học Màu sắc (Cooling Rate):** TDE có đặc trưng "nguội đi" theo thời gian. Chúng tôi tính toán **độ dốc thay đổi màu ($g-r$)** trong khoảng thời gian 20 ngày sau đỉnh sáng để bắt lấy đặc điểm nhiệt động lực học này.
* **⚡ Tối ưu hóa GPU:** Mã nguồn hỗ trợ `torch` và cấu hình XGBoost/CatBoost chạy trên GPU, tăng tốc độ huấn luyện đáng kể.
* **⚖️ Xử lý Mất cân bằng thông minh:** Thay vì sinh dữ liệu giả (SMOTE), chúng tôi sử dụng **Cost-Sensitive Learning** (Học nhạy cảm chi phí) và **Ngưỡng động (Dynamic Thresholding)**.

---

## 🛠 Phương pháp Kỹ thuật

### 1. Feature Engineering (Trích xuất đặc trưng song song)
Quy trình xử lý dữ liệu tích hợp kiến thức miền (Domain Knowledge):

* **Mô hình hóa Bazin (Bazin Fitting):** Khớp đường cong ánh sáng vào hàm Bazin $F(t)$ để trích xuất tham số hình học ($t_{rise}, t_{fall}$), giúp nhận diện hình dạng "tăng nhanh, giảm từ từ" của TDE.
* **Gaussian Process Regression (GP):** Nội suy dữ liệu để dự đoán chính xác Flux tại các thời điểm quan trọng (Peak và Post-Peak).
* **Biến đổi Vật lý:**
    * Tính khoảng cách độ sáng (Luminosity Distance $d_L$).
    * Chuyển đổi $Flux \rightarrow M_{abs}$ (Absolute Magnitude).
* **Thống kê:** Chỉ số Stetson $J, K$ để lọc nhiễu nền.

### 2. Kiến trúc Ensemble Learning
Sử dụng **Soft Voting Classifier** kết hợp 3 mô hình SOTA được tối ưu hóa bằng **Optuna**:

| Model | Cấu hình & Vai trò |
| :--- | :--- |
| **LightGBM** | `boosting_type='dart'` (Dropout Regularization) giúp chống Overfitting. |
| **XGBoost** | `tree_method='hist'` + **GPU Acceleration** xử lý dữ liệu lớn tốc độ cao. |
| **CatBoost** | `auto_class_weights='SqrtBalanced'` tự động cân bằng lớp dữ liệu. |

---

## 📊 Hiệu suất Mô hình

Mô hình được đánh giá nghiêm ngặt qua chiến lược **Stratified 5-Fold Cross-Validation**:

* **F1-Score (Validation):** Đạt mức **~0.6595**, cải thiện rõ rệt so với Baseline nhờ các đặc trưng vật lý mới.
* **Dynamic Thresholding:**
    * Ngưỡng mặc định 0.5 bỏ sót hầu hết TDE.
    * Ngưỡng tối ưu **0.1195** (dựa trên phân vị xác suất) giúp bắt được **382** sự kiện TDE trên tập Test (tương ứng **5.35%**), khớp với tỷ lệ thực tế tự nhiên.

---

## 📂 Cấu trúc Repository

Mã nguồn được tổ chức khoa học:

* **`improved-model-bonus-features.ipynb`**: 🌟 **(Recommended)** Phiên bản cao cấp nhất. Chứa toàn bộ quy trình xử lý đặc trưng vật lý, tính toán độ sáng tuyệt đối, độ dốc màu và tối ưu hóa GPU.
     * File dự đoán cuối cùng: `submission_final_physics.csv`.
* **`improved_model.ipynb`**: Phiên bản ổn định (Stable release), tập trung vào các đặc trưng hình học Bazin và tối ưu tham số cơ bản.

---
