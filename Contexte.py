import streamlit as st

st.subheader('Contexte')



def Contexte():
 st.title("Contexte")   

 from PIL import Image
 
 st.write("Un changement climatique, avec un réchauffement de +1,7°C en France depuis les années 1850, pourrait atteindre +3°C ici 2100 si aucune mesure n'est prise.")
 
 
 if st.checkbox("**Les causes**") :
  st.write('•	Les émissions de gaz à effet de serre (GES) tels que le CO2, le CH4 et le N2O. ')
  st.write('•	L’augmentation de la combustion des énergies fossiles (gaz, pétrole, charbon)')  
  st.write('Le secteur des transports : Fort contributeurs des GES - 39% en France et 23% en Europe.')
 
  from PIL import Image
  image = Image.open("Image1.jpg")

  image = image.resize((600, 400))  

  st.image(image)
 
 if st.checkbox('**Les objectifs**'):
  st.write('•	Identifier les caractéristiques techniques des véhicules qui influent sur les émissions de CO2')
  st.write('•	Prédire les émissions notamment pour les nouveaux modèles produit par les constructeurs.')
 
 if st.checkbox('**Le choix du set de données**'):

  st.write('-   Données du dataset 2013: trop éloignées de la réalité actuelle 2013')
 
  image = Image.open("Image2.jpg")

  image = image.resize((600, 350))  

  st.image(image)
 
  st.write('-   Données du dataset 2019 : changement de norme en 2019 des mesures des émissions de CO2 ')
 
  image = Image.open("Image3.jpg")

  image = image.resize((600, 350))  

  st.image(image)
 
  st.write('Aussi, après plusieurs recherches et démarches nous avons optés pour un dataset récent 2022 plus actuel.')
 
 
