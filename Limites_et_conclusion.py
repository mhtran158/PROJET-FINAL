import streamlit as st

st.subheader('Limites et conclusion')

import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from PIL import Image



def Limites_et_conclusion():
  st.title("Limites et Conclusion")   
  
  image = Image.open("Imageconclu.jpg")

  image = image.resize((700, 300))  

  st.image(image)  
  
  st.subheader("Les Limites")
  if st.checkbox("**Caractéristiques techniques et technologiques des véhicules**") :
     image = Image.open("tt.jpg")

     image = image.resize((400, 300))  

     st.image(image)     

   
  if st.checkbox("**Caractéristiques relatives à l’usage**") :
     image = Image.open("tu.jpg")

     image = image.resize((450, 200))  

     st.image(image)     

    
      
  st.subheader('Conclusion')  
  
  if st.checkbox("**Les émissions de CO2 liées à l’usage des véhicules**") :
      
    image = Image.open("Image13.jpg")

    image = image.resize((900, 350))  

    st.image(image)   
 

    image = Image.open("Image14.jpg")

    image = image.resize((700, 350))  

    st.image(image)   
   
  if st.checkbox("**Les émissions de CO2 de l’industrie automobile**") :
 
    
   image = Image.open("Image15.jpg")

   image = image.resize((700, 450))  

   st.image(image)   
  
  

  