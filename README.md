Chest X-Ray Disease Detection System with GUI

📌 Project Overview
This project presents an end-to-end chest X-ray disease detection system that classifies images into COVID-19, Pneumonia, and Normal categories using a custom CNN model, followed by lung segmentation and a Tkinter-based GUI for user interaction.

📂 Dataset
Chest X-Ray Images Dataset
Classes: COVID, Pneumonia, Normal
Image size: 256×256
Data split: 80% training, 20% testing
🛠️ Preprocessing & Segmentation
Grayscale conversion and normalization
Image resizing to 256×256

🧠 Model
Custom CNN (no pretrained models)
Conv2D + MaxPooling layers
Dense + Dropout layers
Softmax output for multi-class classification

⚙️ Training
Optimizer: Adam
Loss: Categorical Crossentropy
Metric: AUC
Class imbalance handled using class weights

🖥️ GUI (Tkinter)
User can upload chest X-ray image
Displays:
Predicted disease class
Confidence score
Designed as a desktop medical screening tool

📊 Results
Validation AUC ≈ 99%
Low validation loss
Accurate predictions on unseen images

✅ Conclusion
The system integrates classification, segmentation, and a user-friendly GUI, making it suitable for clinical assistance and academic demonstration without relying on pretrained models.

Dataset Citation:

This dataset is taken from,

Kumar, Sachin (2022), “Covid19-Pneumonia-Normal Chest X-Ray Images”, Mendeley Data, V1, doi: 10.17632/dvntn9yhd2.1
