# StockPredict — Dự báo giá cổ phiếu bằng Machine Learning

## Mục tiêu
Dự báo **giá đóng cửa ngày tiếp theo** của cổ phiếu Việt Nam dựa trên dữ liệu lịch sử, sử dụng các thuật toán Machine Learning.

---

## Giải pháp
- Thu thập dữ liệu cổ phiếu từ thư viện **vnstock**.  
- Tạo đặc trưng (feature engineering):  
  - `ma_5`, `ma_10`: trung bình động 5 & 10 ngày  
  - `volatility`: độ biến động giá  
  - `momentum`: đà tăng/giảm  
  - `target`: giá đóng cửa ngày tiếp theo  
- Chia dữ liệu train/test theo thời gian.  
- So sánh 4 mô hình hồi quy:  
  - Linear Regression  
  - Random Forest Regressor  
  - XGBoost Regressor  
  - CatBoost Regressor  
- Đánh giá bằng: RMSE, MAE, R², Cross-Validation (cv=5).

 Cách sử dụng  
1. Cài đặt các thư viện cần thiết (ví dụ: `vnstock`, `pandas`, `scikit-learn`, `xgboost`, `catboost`, `matplotlib`).  
2. Mở và chạy notebook `stock.ipynb` trong môi trường Jupyter, tuần tự từ đầu đến cuối.  
3. Xem các kết quả đánh giá: so sánh hiệu suất các mô hình và xác định mô hình tốt nhất.  
4. Tùy chỉnh hoặc mở rộng notebook theo nhu cầu: thêm feature, validation theo chuỗi thời gian, hoặc thử mô hình mới. 

---

## Trainning

```python
import pandas as pd
from vnstock import * 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Tải dữ liệu cổ phiếu
df = stock_historical_data('VCB', '2024-01-01', '2025-01-01')

# Tạo đặc trưng
df['ma_5'] = df['close'].rolling(5).mean()
df['ma_10'] = df['close'].rolling(10).mean()
df['volatility'] = (df['high'] - df['low']) / df['low']
df['momentum'] = df['close'] / df['close'].shift(4) - 1
df['target'] = df['close'].shift(-1)
df.dropna(inplace=True)

# Chia dữ liệu
X = df[['ma_5', 'ma_10', 'volatility', 'momentum']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Huấn luyện mô hình
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Đánh giá
print('MAE:', mean_absolute_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R²:', r2_score(y_test, y_pred))
```
