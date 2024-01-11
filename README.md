# Diabetes Prediction Web App

A Diabetes Prediction web app using machine learning is an application that utilizes a trained machine learning model to predict whether a person has diabetes based on input features such as glucose levels, blood pressure, BMI, age, etc

## Prerequisites

- Python
- Pip 
- Git (optional)

## Getting Started

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/shubham5027/diabetes-prediction-webapp.git
    cd diabetes-prediction-webapp
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the App:**

    ```bash
    python -m streamlit run app.py
    ```

4. **Access the Web App:**

![Screenshot 2024-01-11 144256](https://github.com/shubham5027/Diabetes-Prediction-webapp-using-Machine-Learning/assets/132193443/1cf49525-f715-412f-b076-de9f0d4de095)
![Screenshot 2024-01-11 144308](https://github.com/shubham5027/Diabetes-Prediction-webapp-using-Machine-Learning/assets/132193443/63b77e4a-7c5b-4f4c-a7f0-3127947cdceb)



## Usage

- Use the sliders in the sidebar to input values for features like pregnancies, glucose, blood pressure, etc.
- Click the "Predict" button to see the model's prediction.
- The result will be displayed, indicating whether the person is predicted to have diabetes or not.

## Model Information

The machine learning model used in this app is a logistic regression model trained on the Diabetes dataset.

## Additional Notes

- The web app is built using Streamlit, a Python library for creating web applications with minimal effort.
- The trained machine learning model is saved in a file named `your_model.pkl`. Replace it with your actual trained model file.

## License

This project is licensed under the [MIT License](LICENSE).

