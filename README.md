# Optimization-of-Tennis-Skill-Teaching-and-Performance-Prediction-by-BPNN

This repository contains the source code, data generation scripts, and model implementations for the research paper: **"Optimization of Tennis Skill Teaching and Performance Prediction by Artificial Intelligence and Backpropagation Neural Network" (2025)**.

## Project Overview

This study constructs a Backpropagation Neural Network (BPNN) to predict the competitive performance of adolescent tennis players based on multi-dimensional data (Technical, Physical, and Physiological). The model utilizes a chronological data split to validate its ability to monitor longitudinal skill development.

### Key Features
*   **Model:** 3-Layer BPNN (Input-Hidden-Output) with Sigmoid activation and L2 regularization.
*   **Optimization:** Stochastic Gradient Descent (SGD) with Momentum.
*   **Interpretability:** Implementation of Garson's Algorithm to analyze connection weights and SHAP value analysis.
*   **Baselines:** Comparison with Linear Regression, SVR, Random Forest, and XGBoost.

## Repository Structure

*   `main.py`: The primary script to generate data, train models, and output evaluation metrics.
*   `models.py`: Keras implementation of BPNN and Scikit-learn wrappers for baseline models.
*   `data_utils.py`: Synthetic data generator (mimicking the statistical properties of the private dataset) and preprocessing pipeline.
*   `analysis.py`: Tools for model interpretability (Garson's Algorithm, SHAP).

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
