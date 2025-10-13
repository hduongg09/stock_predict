Stock predict
Mục tiêu:
Dự đoán giá cổ phiếu Việt Nam dựa trên dữ liệu lịch sử
Dùng các mô hình Machine Learning (Linear Regression, Random Forest, XGBoost, CatBoost) để học dữ liệu giá quá khứ và dự đoán giá đóng cửa cho ngày tiếp theo

các bước:
- Sử dụng thư viện vnstock để tải dữ liệu giá cổ phiếu Việt Nam
- Làm sạch dữ liệu
- Thống kê mô tả nhanh (EDA): giá trị trung bình, độ lệch chuẩn, khoảng thời gian dữ liệu, số mã.
- Tạo các đặc trưng (Feature Engineering) :
  - 'ma_5', 'ma_10' :Trung bình 5 và 10 ngày
  - `volatility`: độ biến động giá theo `(high - low) / low`  
  - `momentum`: đà tăng/giảm của cổ phiếu (`close_t / close_(t-4) - 1`)  
  - `target`: giá đóng cửa của ngày tiếp theo (`shift(-1)`)
- So sánh hiệu năng 4 thuật toán hồi quy:
 - **Linear Regression**
 - **Random Forest Regressor**
 - **XGBoost Regressor**
 - **CatBoost Regressor**

- Huấn luyện và đánh giá bằng:
 - **Cross-validation (cv=5)**  
 - Các chỉ số đánh giá:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (Hệ số xác định)
