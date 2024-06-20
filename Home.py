import streamlit as st

def Home():
    
    st.title("Projet: Modelisation des Emissions de CO2")
    st.subheader('Auteurs:')
    st.write("Ayaovi DJADJAGLO-AMEGEE")
    st.write("Minh TRAN")
    st.write("Nathalie GUERRIN")
    st.write("Nicolas BEAUDRON")
    
    from PIL import Image
   
    image = Image.open("Image12.jpg")

    image = image.resize((750, 400))  

    st.image(image) 


