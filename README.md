# Dog Breed Identification

This repository contains my solution notebook for the [Dog Breed Identification competition](https://www.kaggle.com/c/dog-breed-identification) on Kaggle.  
The goal of the competition is to classify images of dogs into **120 different breeds** using machine learning and deep learning models.

---

## Project Structure
- `dog_breed_id.ipynb` – main notebook with data exploration, preprocessing, model building, training, and evaluation.  
- `full_model_predictions_resnetv2_1.csv` – my prediction file submitted to Kaggle.
This submission achieved a **log loss score of 1.02864** on the public leaderboard

---

## Dataset
**Source:** [Kaggle Dog Breed Identification Dataset](https://www.kaggle.com/c/dog-breed-identification/data)  

---

## Approach
- Preprocessing and augmentation of input images.  
- Transfer learning with **ResNetV2 feature extractors** ResNetV2-50 from Kaggle.  
- Added classification head with softmax activation.  
- Trained with **early stopping** and **TensorBoard monitoring**.  
- Evaluated using log loss and accuracy.  

---

## Requirements
This notebook was developed in **Google Colab** using **TensorFlow 2.x**.  

Install dependencies (if running locally):  
```bash
pip install tensorflow tensorflow-hub matplotlib pandas numpy
```
