# Dash Application For Thai Numbers Recognition

## Objective
โปรเจ็คนี้เป็นโปรเจ็คที่จัดทำขึ้นเพื่อส่งในวิชา Data Aanalytics and Data Science Tools and Programming ของสถาบันบัณฑิตพัฒนบริหารศาสตร์(NIDA) โดยวัตถุประสงค์ของโปรเจ็คนี้ คือการสร้าง Dash Application เพื่อให้สามารถสร้าง 'Machine Learning Model สำหรับจดจำและทำนายตัวเลขภาษาไทยที่เขียนด้วยลายมือ' ผ่านทาง website ได้ 

โดย code ส่วนของการสร้าง model เราจะอ้างอิงจากโปรเจ็ค 'Thai Number Recognition' ที่เราได้ทำขึ้นก่อนหน้านี้

ซึ่งเว็บไซต์ หรือผลลัพธ์ที่ได้นั้นจะมีทั้งหมด 4 ส่วน นั่นคือ

  1. Input Data for Training
  2. Model Result and Evaluation
  3. Input Image for Testing
  4. Function for Testing Number

## 1. Input Data for Training

เริ่มต้น เราได้ทำ website ส่วนของการรับ input หรือรูปภาพที่จะทำการ train โดยได้สร้างวิธีการรับ input 2 ส่วน คือ

  1. รับ input รูปแบบ image
  2. รับ input รูปแบบ CSV

ซึ่ง user สามารถเลือกได้ตามสะดวกว่าต้องการ input data เข้า website ด้วยวิธีไหน

### 1.1 Data for Training (Image)

ในส่วนนี้ user สามารถส่งรูปที่ต้องการนำไป train ทีละตัวเลขได้

![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/593d486b-b7bb-4c1f-940c-a764cbf91a8d)

![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/900afbc1-4617-4c33-850f-45abf2aee9e9)

### 1.2 Data for Training (CSV)

รวมถึง user สามารถนำไฟล์ CSV (ซึ่งสามารถสร้างได้จากการการรัน code ของโปรเจ็ค 'Thai Number Recognition' ที่เคยจัดทำขึ้นก่อนหน้านี้) ใส่ลงใน website ได้

![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/1f0e1d60-26e4-4c95-bc5a-d634b9d0f186)

## 2. Model Result and Evaluation

เมื่อ user ใส่ input เสร็จแล้ว สามารถกด submit เพื่อให้ website เริ่มต้นสร้าง model

![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/656f4689-1c5f-40c1-8a6c-c12a5d6e1758)

โดยเมื่อกด submit แล้ว website จะใช้เวลาประมาณ 3-5 นาที ในการสร้างและหา model ที่เหมาะสมที่สุดสำหรับ data ที่ user ใส่เข้าไป แล้วเมื่อสร้างเสร็จ website จะแสดง evaluation ของ model ที่สร้างขึ้น

![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/3a063ec2-b2a7-4a7c-991d-a77515a9787d)
![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/48c0fef5-e011-4e83-a7b1-2ddbdb348ce9)
![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/973c5544-f740-44fd-808d-950263d52f20)

โดยจากผลลัพธ์ที่ได้จากตัวอย่างนี้ สามารถบอกได้ว่า
  1. จำนวน data ทั้งหมดที่ user ใส่เข้าไป คือ 400 หน่วย
  2. จำนวน data ที่นำไป train model คือ 320 หน่วย
  3. จำนวน data สำหรับการ test model คือ 80 หน่วย
  4. website ได้ทำการสร้าง model ทั้งหมด 3 model โดย model ที่ดีที่สุดคือ Extra Trees Classifier ซึ่งมีค่า Accuracy อยู่ที่ 0.9512
  5. บ่งบอกประสิทธิภาพของ model ด้วย ROC Curve และ Confusion Metrix
  6. บอกค่า Accuracy ของ Train set ซึ่งมีค่าอยู่ที่ 0.9875
  7. บอกค่า Precision ของ Train set ซึ่งมีค่าอยู่ที่ 0.988
  8. บอกค่า Recall ของ Train set ซึ่งมีค่าอยู่ที่ 0.9875

รวมถึงจะแสดงผลลัพธ์การ Prediction จาก test data โดยได้ผลลัพธ์เป็นดังนี้

![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/c895970d-cd71-4053-a731-65c40d85685a)
![image](https://github.com/MeenWhile/ML-Image-Thai-Numbers-Recognition/assets/125643589/3087c63a-f0b1-4ed6-af4c-e7fe64c950ba)

โดยจากผลลัพธ์ที่ได้จาก test data นี้ สามารถบอกได้ว่า
  1. บ่งบอกประสิทธิภาพของการ prediction ด้วย ROC Curve และ Confusion Metrix
  2. บอกค่า Accuracy ของ Test set ซึ่งมีค่าอยู่ที่ 0.95
  3. บอกค่า Precision ของ Test set ซึ่งมีค่าอยู่ที่ 0.958
  4. บอกค่า Recall ของ Test set ซึ่งมีค่าอยู่ที่ 0.95
