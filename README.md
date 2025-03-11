# Predictive Maintenance Model

## ✨ เกี่ยวกับโปรเจค
บริษัทมีอุปกรณ์ที่ส่งข้อมูลเซ็นเซอร์รายวัน และต้องการพัฒนาโซลูชันการบำรุงรักษาเชิงพยากรณ์ (Predictive Maintenance) เพื่อระบุเวลาที่ควรดำเนินการซ่อมบำรุง ช่วยลดต้นทุนเมื่อเทียบกับการบำรุงรักษาตามรอบเวลา

โปรเจคนี้มีเป้าหมายในการสร้างโมเดล Machine Learning เพื่อทำนายความน่าจะเป็นที่อุปกรณ์จะล้มเหลว โดยให้ความสำคัญกับการลด False Positives และ False Negatives 

## 📂 Dataset
- **แหล่งที่มา**: ข้อมูลที่รวบรวมจากเซ็นเซอร์ของอุปกรณ์
- **เป้าหมาย**: ทำนายค่าคอลัมน์ `failure` (0 = ไม่ล้มเหลว, 1 = ล้มเหลว)
- **ข้อมูลที่มี**:
  | Column Name | Description |
  |------------|-------------|
  | `date`     | Date of record |
  | `device`   | Device identifier |
  | `failure`  | Failure occurrence (e.g., 0 = No failure, 1 = Failure) |
  | `metric1` - `metric9` | Various performance metrics |

## 📊 กระบวนการพัฒนาโมเดล
1. **Data Cleaning**: ตรวจสอบค่าสูญหาย และปรับปรุงคุณภาพข้อมูล
2. **Exploratory Data Analysis (EDA)**: วิเคราะห์ข้อมูลผ่านการสร้างกราฟ เช่น Histogram, Heatmap
3. **Feature Selection & Engineering:**: เลือกฟีเจอร์ที่สำคัญต่อการพยากรณ์ และใช้เทคนิค Feature Engineering
4. **Machine Learning Model Development**: ใช้อัลกอริธึมที่เหมาะสม เช่น Logistic Regression, Random Forest
5. **Model Evaluation**: ใช้เมตริก เช่น Confusion Matrix, Precision, Recall, ROC Curve

## 📊 ตัวอย่างการแสดงผลข้อมูล
### Active Devices per Month
![Failure Distribution](https://raw.githubusercontent.com/Pariman1419/Predictive-Maintenance/main/Active Devices per Month.png)

### Heat map 
![Failure Distribution](https://raw.githubusercontent.com/Pariman1419/Predictive-Maintenance/main/Heatmap.png)


### Distribution of Failure
![Failure Distribution](https://raw.githubusercontent.com/Pariman1419/Predictive-Maintenance/main/FailureDistribution.png)

- ข้อมูลเป็นข้อมูลแบบ Imbalance จึงจำเป็นต้องทำการ SMOTE ก่อนทำการ Train model

## 🛠️ ตัวอย่างโค้ดใช้งานโมเดล
```python

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# แยก Features และ Target
X = df_model.drop(columns=['failure'])
y = df_model['failure']

# ใช้ SMOTE เพื่อเพิ่มตัวอย่างของ class ที่มีจำนวนน้อย
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# แบ่งชุดข้อมูล (X คือ Features, y คือ Target)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, shuffle=True, 
                                                    random_state=42)

# ใช้โมเดลที่ดีที่สุดจาก Grid Search
best_logreg = grid_search_logreg.best_estimator_
best_tree = grid_search_tree.best_estimator_
best_rf = grid_search_rf.best_estimator_

# ทำนายผล
y_pred_logreg = best_logreg.predict(X_test)
y_pred_tree = best_tree.predict(X_test)
y_pred_rf = best_rf.predict(X_test)

# แสดงผลลัพธ์
print("Logistic Regression:\n", classification_report(y_test, y_pred_logreg))
print("Decision Tree:\n", classification_report(y_test, y_pred_tree))
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

```
### ROC Curve 

![Failure Distribution](https://raw.githubusercontent.com/Pariman1419/Predictive-Maintenance/main/ROC.png)

# Performance Metrics

### Logistic Regression
```
Precision: 0.71 | Recall: 0.87 | F1-score: 0.78 | Accuracy: 0.76
```

### Decision Tree
```
Precision: 1.00 | Recall: 1.00 | F1-score: 1.00 | Accuracy: 1.00
```

### Random Forest
```
Precision: 1.00 | Recall: 1.00 | F1-score: 1.00 | Accuracy: 1.00
```

