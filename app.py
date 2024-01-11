import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('diabetes.csv')
diabetes_mean_df = df.groupby('Outcome').mean()


X = df.drop('Outcome',axis=1)
y = df['Outcome']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model=LogisticRegression()
model.fit(X_train, y_train)


train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)


def app():


    st.title('Diabetes Prediction')

    # input form 
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    input_data = [[preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]]
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)

    prediction = None 

    if st.button('Predict'):
        prediction = model.predict(input_data)

    st.write('Based on the input features, the model predicts:')
    if prediction is not None:
        if prediction == 1:
            st.warning('This person has diabetes.')
        else:
            st.success('This person does not have diabetes.')

  
    st.header('Dataset Summary')
    st.write(df.describe())


if __name__ == '__main__':
    app()
