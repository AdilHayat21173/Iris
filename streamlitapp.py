import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
  
# Loading the model to predict on the data 
pickle_in = open('Save Model/classifier_iris.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

  

def prediction(sepal_length, sepal_width, petal_length, petal_width):   
   
    prediction = classifier.predict( 
        [[sepal_length, sepal_width, petal_length, petal_width]]) 
    print(prediction) 
    return prediction 
      
  

def main(): 
      # giving the webpage a title 
    st.title("Iris Flower Prediction") 
      
    sepal_length = st.text_input("Sepal Length") 
    sepal_width = st.text_input("Sepal Width") 
    petal_length = st.text_input("Petal Length") 
    petal_width = st.text_input("Petal Width") 
    result ="" 
      
   
    if st.button("Predict"): 
        result = prediction(sepal_length, sepal_width, petal_length, petal_width) 
    st.success('The output is {}'.format(result)) 
     
if __name__=='__main__': 
    main() 