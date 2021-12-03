import numpy as np
import pickle
from numpy.core.fromnumeric import diagonal
import streamlit as st


#loading model
loaded_model = pickle.load(open('trained_model.sav' , 'rb'))

#prediction function
def heart_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'


# heart_prediction([52,1,2,172,199,1,1,162,0,0.5,2,0,3])

def main():
    st.title('Heart Disease Prediction')

    age = st.text_input('Age')
    sex = st.text_input('sex (Male = 1 , Female = 0)')
    cp = st.text_input('Chest Pain Type')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestrol')
    fbs = st.text_input('Fasting Blood Sugar')
    restecg = st.text_input('Resting ECG')
    thalach = st.text_input('Max Heart Rate')
    exang = st.text_input('Exercise iclude angina')
    oldpeak = st.text_input('oldpeak')
    slope = st.text_input('Peak Exercie')
    ca = st.text_input('Major Vessels')
    thal = st.text_input('thalassemia')
    


    #code for prediction
    diagnosis = ''

    if st.button('Test Result'):
        diagnosis = heart_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

        st.success(diagnosis)


if __name__ == "__main__":
    main()

