<h1>Autism Video Classification using LSTM and BiLSTM </h1>

  <h2>📌 Overview </h2>
  
This project focuses on classifying videos for autism-related behavioral patterns using Long Short-Term Memory (LSTM) and Bidirectional LSTM (BiLSTM) deep learning architectures.
The aim is to leverage temporal sequence modeling to detect subtle patterns in movements, facial expressions, or behaviors from video sequences that could indicate autism spectrum disorder (ASD).

This work was carried out as part of a research internship at National Institute of Technology Tiruchirappalli (NIT Trichy) under the guidance of Prof. Dr. Varun P. Gopi and mentor Sreeraj Sahadevan.

<h2>🎯 Objectives<h2></h2>
To develop a deep learning-based video classification pipeline.

To compare the performance of LSTM and BiLSTM models in behavioral pattern recognition.

To contribute towards early screening tools for autism using AI.

<h2>📂 Dataset</h2>

Structure:

Training videos: 70

Testing videos: 30

<h3>Preprocessing:</h3>

Frame extraction at fixed intervals

Resizing and normalization

Feature extraction using CNN 

<h3>⚙️ Methodology</h3>
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
