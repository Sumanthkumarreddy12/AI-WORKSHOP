# Heart Disease Prediction Web App

## Project Overview

This project builds a Machine Learning model to predict whether a patient has heart disease based on clinical measurements.
The application is deployed as a simple web interface using Streamlit where users can enter patient details and get predictions.

## Problem Statement

Build a model to predict whether a patient has heart disease based on clinical measurements.

Target Variable:

* 0 → No Heart Disease
* 1 → Heart Disease

## Key Features

* Age
* Sex
* Cholesterol
* Resting Blood Pressure
* Maximum Heart Rate
* Chest Pain Type

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* NumPy

## Project Structure

```
HeartDiseaseProject
│
├── app.py
├── heart.csv
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository or download the project.

2. Install required libraries:

```
pip install -r requirements.txt
```

3. Run the application:

```
streamlit run app.py
```

4. Open the browser and go to:

```
http://localhost:8501
```

## Model Used

Random Forest Classifier is used to train the prediction model.

## Output

The application predicts whether a patient has heart disease or not.

* 0 → No Heart Disease
* 1 → Heart Disease

## Future Improvements

* Add more clinical features
* Improve model accuracy
* Deploy the application online
