# 🎙️ Speaker Diarization and Emotion Recognition

This repository contains the implementation for a machine learning framework that integrates **speaker diarization** using **PyAnnote Audio** and **emotion classification** using a **CNN-based model**.

## 🧠 Key Features

- **Speaker Diarization**: Identifies "who spoke when" using PyAnnote Audio.
- **Emotion Recognition**: Classifies emotions from speech signals with a CNN model.
- **Datasets Used**:  
  - RAVDESS  
  - CREMA-D  
  - TESS  
  - SAVEE

## 📈 Highlights

- Extracted key audio features from WAV files.
- Achieved **63% accuracy** in classifying emotions.
- Supports end-to-end processing from raw audio to emotion labels.

## 📁 Structure

- `datasets/`: Organized audio files from multiple datasets.
- `outputs/`: Contains trained model weights and encoders.
- `run.py`: Main script to execute the pipeline.
- `requirements.txt`: Dependencies for running the project.

## 📄 Reference

This work is supported by our paper:  
[*Emotion Recognition from Speech using Deep Learning Models*](https://arxiv.org/abs/2310.12851)
