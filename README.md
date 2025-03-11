# Predictive Maintenance Model

## ✨ เกี่ยวกับโปรเจค
บริษัทมีอุปกรณ์ที่ส่งข้อมูลเซ็นเซอร์รายวัน และต้องการพัฒนาโซลูชันการบำรุงรักษาเชิงพยากรณ์ (Predictive Maintenance) เพื่อระบุเวลาที่ควรดำเนินการซ่อมบำรุง ช่วยลดต้นทุนเมื่อเทียบกับการบำรุงรักษาตามรอบเวลา

โปรเจคนี้มีเป้าหมายในการสร้างโมเดล Machine Learning เพื่อทำนายความน่าจะเป็นที่อุปกรณ์จะล้มเหลว โดยให้ความสำคัญกับการลด False Positives และ False Negatives ซึ่งเป็นปัจจัยที่สำคัญในกระบวนการตัดสินใจของธุรกิจ

## 📂 Dataset
- **แหล่งที่มา**: [Predictive Maintenance Dataset](ttps://www.kaggle.com/datasets/hiimanshuagarwal/predictive-maintenance-dataset/data) 
- **เป้าหมาย**: ทำนายค่าคอลัมน์ `failure` (0 = ไม่ล้มเหลว, 1 = ล้มเหลว)
- **ข้อมูลที่มี**:
  | Column Name | Description |
  |------------|-------------|
  | `date`     | Date of record |
  | `device`   | Device identifier |
  | `failure`  | Failure occurrence (e.g., 0 = No failure, 1 = Failure) |
  | `metric1` - `metric9` | Various performance metrics |
  

## 📊 กระบวนการพัฒนาโมเดล
1. **Data Cleaning**: ตรวจสอบค่าสูญหาย และปรับปรุงคุณภาพข้อมูล เช่น การเติมค่าสูญหาย การปรับขนาดข้อมูล
2. **Exploratory Data Analysis (EDA)**: วิเคราะห์ข้อมูลเบื้องต้นผ่านการสร้างกราฟ เช่น Histogram, Heatmap เพื่อดูความสัมพันธ์และรูปแบบของข้อมูล
3. **Feature Selection & Engineering:**: เเลือกฟีเจอร์ที่สำคัญต่อการพยากรณ์ และใช้เทคนิค Feature Engineering เช่น การแปลงข้อมูลเชิงสถิติ หรือการเพิ่มฟีเจอร์ใหม่จากข้อมูลที่มี
4. **Machine Learning Model Development**: ใช้อัลกอริธึมที่เหมาะสม เช่น Logistic Regression, Random Forest, Decision Tree เพื่อสร้างโมเดลพยากรณ์
5. **Model Evaluation**: ใช้เมตริกต่าง ๆ เช่น Confusion Matrix, Precision, Recall, ROC Curve เพื่อประเมินประสิทธิภาพของโมเดล

## 📊 ตัวอย่างการแสดงผลข้อมูล


### 📱📅 Active Devices per Month
<img src="https://raw.githubusercontent.com/Pariman1419/Predictive-Maintenance/main/Active%20Devices%20per%20Month.png" width="500">

- แสดงจำนวนอุปกรณ์ที่ใช้งานในแต่ละเดือน ช่วยติดตามแนวโน้มการใช้งานและการบำรุงรักษาอุปกรณ์
  

### 🌡️🗺️ Heat map 
<img src="https://raw.githubusercontent.com/Pariman1419/Predictive-Maintenance/main/Heatmap.png" width="500">

- แสดงความสัมพันธ์ระหว่างฟีเจอร์ต่าง ๆ กับการล้มเหลวของอุปกรณ์ ใช้สีในการแสดงความเข้มข้นของความสัมพันธ์
  

### ⚠️📊 Distribution of Failure
<img src="https://raw.githubusercontent.com/Pariman1419/Predictive-Maintenance/main/FailureDistribution.png" width="500">

- แสดงการกระจายของข้อมูลการล้มเหลว (0 = ไม่ล้มเหลว, 1 = ล้มเหลว) เพื่อเข้าใจข้อมูลที่ไม่สมดุล
- ข้อมูลเป็นข้อมูลแบบ Imbalance จึงจำเป็นต้องทำการ SMOTE (Synthetic Minority Over-sampling Technique) เพื่อเพิ่มจำนวนข้อมูลของ class ที่มีจำนวนน้อยก่อนทำการ Train model
  

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

### 📉🔄 ROC Curve 

<img src="https://raw.githubusercontent.com/Pariman1419/Predictive-Maintenance/main/ROC.png" width="500">

- แสดงประสิทธิภาพของโมเดลการจำแนกประเภท โดยเปรียบเทียบระหว่าง True Positive Rate และ False Positive Rate
  

# 📈📊 Performance Metrics

### แสดงเมตริกต่าง ๆ เช่น Precision, Recall, F1-Score และ Accuracy เพื่อประเมินความแม่นยำของโมเดล

- Precision: บอกว่าในทั้งหมดที่โมเดลทำนายว่าเป็น failure, กี่เปอร์เซ็นต์ที่เป็นจริง
- Recall: บอกว่าในทั้งหมดที่เป็นจริง failure, โมเดลสามารถทำนายได้ถูกต้องกี่เปอร์เซ็นต์
- F1-Score: คำนวณจากค่า Precision และ Recall เพื่อให้ได้ผลลัพธ์ที่สมดุล
- Accuracy: ความแม่นยำโดยรวมของโมเดลในการทำนายทั้งสองคลาส
  
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

