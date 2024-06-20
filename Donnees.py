import streamlit as st

st.subheader('Exploration des données et nettoyage')


 

def Donnees():
 st.title("Données et nettoyage des données")
 
  
 from PIL import Image
 image = Image.open("Image4.jpg")

 image = image.resize((1500, 500))  

 st.image(image)
 
 
 st.write('Pour faciliter la lecture, nous renommons les variables à l’aide d’un dictionnaire qui donne les correspondances suivantes:')
 st.write(' •	m (kg):		masse')
 st.write(' •	Ewltp (kg):C02')
 st.write(' •	W (mm):		dimension')
 st.write(' •	ec (cm3):	cylindrée')
 st.write(' •	ep (KW):	puissance')
 st.write(' •	Electric range(km):	autonomie électrique')
 st.write(' •	Fm:	fuel mode')
 st.write(' •	Ft:	fuel type')
 
 
 st.write('Nous identifions et gérons les doublons du dataset data_target.')
 st.write('Il comporte 9 392 502 doublons que nous supprimons, ce qui amène le dataset à 87 042 lignes à ce stade')
 st.write('Puis Gestion des données manquantes (avec le fichier dans streamlit)')
 
 image = Image.open("Image5.jpg")

 image = image.resize((450, 350))  

 st.image(image)
 
 
 image = Image.open("Image6.jpg")

 image = image.resize((450, 350))  

 st.image(image)
 
 image = Image.open("Image7.jpg")

 image = image.resize((450, 350))  

 st.image(image)
 
 
 image = Image.open("Image8.jpg")

 image = image.resize((450, 350))  

 st.image(image)
 
 image = Image.open("Image9.jpg")

 image = image.resize((450, 350))  

 st.image(image)
 
 image = Image.open("Image10.jpg")

 image = image.resize((450, 350))  

 st.image(image)
 
 image = Image.open("Image11.jpg")

 image = image.resize((450, 350))  

 st.image(image)
 

 
