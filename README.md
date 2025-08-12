Autism Video Classification using LSTM and BiLSTM

📌 Overview
This project focuses on classifying videos for autism-related behavioral patterns using Long Short-Term Memory (LSTM) and Bidirectional LSTM (BiLSTM) deep learning architectures.
The aim is to leverage temporal sequence modeling to detect subtle patterns in movements, facial expressions, or behaviors from video sequences that could indicate autism spectrum disorder (ASD).

This work was carried out as part of a research internship at National Institute of Technology Tiruchirappalli (NIT Trichy) under the guidance of Prof. Dr. Varun P. Gopi and mentor Sreeraj Sahadevan.

🎯 Objectives
To develop a deep learning-based video classification pipeline.

To compare the performance of LSTM and BiLSTM models in behavioral pattern recognition.

To contribute towards early screening tools for autism using AI.

📂 Dataset
Source: (Specify dataset name or institution if possible)

Structure:

Training videos: 70

Testing videos: 30

Preprocessing:

Frame extraction at fixed intervals

Resizing and normalization

Feature extraction using CNN 

⚙️ Methodology
Data Preprocessing

Video-to-frame conversion

Feature extraction using pre-trained CNN

Sequence creation for temporal modeling

Model Development

LSTM architecture

BiLSTM architecture

Dropout and regularization techniques

Training & Evaluation

Loss function: Categorical Crossentropy

Optimizer: Adam

Metrics: Accuracy, Precision, Recall
