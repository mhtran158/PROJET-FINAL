import streamlit as st

st.subheader('Démonstration du modèle retenu')


import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def Demo():
 st.title("Application de prévision des émissions de CO2")    
    
 data_target13 = pd.read_csv('data_target13.csv',sep=',',encoding="latin-1")
 data_target13.head()

 data_target13.info()

# On renomme les variables

 dictionnaire ={'masse':'Poids (kg)',
               'C02': 'CO2',
               'cylindrÃ©e':'Cylindrée (cm3)',
               'puissance':'Puissance (ch)',
               'dimension': 'Dimension (mm)',
               'autonomie Ã©lectrique': 'Autonomie electrique (km)',
               'fuel mode' : 'fuel_mode',
               'fuel type': 'fuel_type'}

 data_target13 = data_target13.rename(dictionnaire, axis = 1)

 data_target13.head() 

 from PIL import Image

#Ouverture de l'image avec PIL
 image = Image.open("photocar.jpg")

# Redimensionnement de l'image
 image = image.resize((650, 450))  # taille désirée (largeur, hauteur)

# Affichage de l'image redimensionnée
 st.image(image)

 st.header("Prédiction de l'émission de C02") 
 st.sidebar.subheader(" Caractéristiques techniques du véhicule")


 colonnes_selectionnees = ['Poids (kg)','Dimension (mm)','Cylindrée (cm3)','Puissance (ch)','Autonomie electrique (km)']


#On partage le dataset en variables explicatives (masse, dimension, cylindrée, puissance, autonomie éléctrique, fuel mode,
#fuel type) et en variable cible :CO2

 feats = data_target13.drop('CO2', axis = 1)
 target = data_target13['CO2'] # Variable cible
 feats.info()

# Séparation du données en jeu d'entrainement et de test de sorte que le jeu de test contienne 20% des données du dataset:


 X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.20, random_state = 42)

# Séparation des colonnes catégorielles des colonnes numériques et création des dataframe correspondants 
# sur les jeux d'entrainement et de test

 num = ['Poids (kg)', 'Cylindrée (cm3)', 'Puissance (ch)', 'Dimension (mm)', 'Autonomie electrique (km)'] # num= liste des variables explicatives numériques
 num_train = X_train[num]
 num_test = X_test[num]

 cat = ['fuel_mode','fuel_type'] # cat = liste des variables explicatives catégorielles
 cat_train = X_train[cat]
 cat_test = X_test[cat]

# Encodage des variables catégorielles

 from sklearn.preprocessing import OneHotEncoder

 encoder = OneHotEncoder() # creation de 'encoder' un objet OneHotEncoder

# fit and transform des colonnes de cat_train columns / création d'un nouveau dataframe à partir d'un numpy array
 one_cat_train_array = encoder.fit_transform(cat_train).toarray() 
 one_cat_train_df = pd.DataFrame(one_cat_train_array, columns=encoder.get_feature_names_out())
 one_cat_train_df.head()

# fit and transform des colonnes de cat_test columns / création d'un nouveau dataframe à partir d'un numpy array
 one_cat_test_array = encoder.transform(cat_test).toarray() 
 one_cat_test_df = pd.DataFrame(one_cat_test_array, columns=encoder.get_feature_names_out())
 one_cat_test_df.head(10)

# Standardisation des colonne numériques

 sc = StandardScaler()
 num_train_scaled_array = sc.fit_transform(num_train)
 num_train_scaled_df = pd.DataFrame(num_train_scaled_array, columns=sc.get_feature_names_out())
 num_test_scaled_array = sc.transform(num_test)
 num_test_scaled_df = pd.DataFrame(num_test_scaled_array, columns=sc.get_feature_names_out())

# Reconstitution des jeux d'entraînement et de test après encodage des variables catégorielles 
# en concaténant num_train avec cat_train et num_test avec cat_test.

 X_train_new = pd.concat([num_train_scaled_df, one_cat_train_df.set_index(num_train_scaled_df.index)], axis=1)
 X_test_new = pd.concat([num_test_scaled_df, one_cat_test_df.set_index(num_test_scaled_df.index)], axis=1)


 def user_input():   
    masse=st.sidebar.number_input('Poids (kg)',min_value=0, max_value=5000,value=1500)
    dimension=st.sidebar.number_input('Dimension (mm)',min_value=0, max_value=4000,value=2000)
    cylindrée=st.sidebar.number_input('Cylindrée (cm3)',min_value=0, max_value=10000,value=2000)
    puissance=st.sidebar.number_input('Puissance (ch)',min_value=0, max_value=1000,value=100)
    autonomie_électrique = st.sidebar.number_input('Autonomie electrique (km)',min_value=0, max_value=110,value=0)
    fuel_mode =st.sidebar.selectbox('fuel_mode',options=['M','H','B','P','F'])
    fuel_type =st.sidebar.selectbox('fuel_type',options=['petrol','diesel','ng','Ipg','petrol/electric','diesel/electric','e85'])

    data={'Poids (kg)':masse,
       'Dimension (mm)':dimension,
       'Cylindrée (cm3)':cylindrée,
       'Puissance (ch)':puissance,
       'Autonomie electrique (km)':autonomie_électrique,
       'fuel_mode':fuel_mode,
       'fuel_type':fuel_type}                         
    caractéristiques=pd.DataFrame(data, index=[0])
    return caractéristiques
  
 input_df=user_input()
 st.subheader("Caractéristiques selectionnées :") 
 st.write(input_df) 
 input_df.info()  

 CO2_input=data_target13.drop(columns=['CO2'],axis=1)
 CO2_input.head()
 donnee_entree=pd.concat([input_df,CO2_input], axis=0)
 donnee_entree.head()
 donnee_entree.info()

#num = ['masse', 'cylindrée', 'puissance', 'dimension', 'autonomie_électrique'] 

 num = ['Poids (kg)', 'Cylindrée (cm3)', 'Puissance (ch)', 'Dimension (mm)', 'Autonomie electrique (km)']

 encoder = OneHotEncoder()
 donnee_entree_encoded=encoder.fit_transform(donnee_entree[['fuel_mode','fuel_type']]).toarray()
 donnee_entree_encoded_df=pd.DataFrame(donnee_entree_encoded,columns=encoder.get_feature_names_out(['fuel_mode','fuel_type']))
 donnee_entree_encoded_df.head()
 donnee_entree_encoded_df.info()
    
 sc = StandardScaler()
 num_scaled = sc.fit_transform(donnee_entree[num])
 num_scaled_df = pd.DataFrame(num_scaled, columns=sc.get_feature_names_out(num))
 num_scaled_df.head()

 donnee_entree_final = pd.concat([num_scaled_df, donnee_entree_encoded_df.set_index(num_scaled_df.index)], axis=1)

 donnee_entree_final.head()
 donnee_entree_final.info()
 finale=donnee_entree_final[:1]

#st.subheader('Les caractéristiques transformés:')
#st.dataframe(finale)    

 random_forest=joblib.load('model_rf')
 y_pred_reg = random_forest.predict(finale)

 st.subheader('Prédiction Emission de CO2 (g/km)')
 resultat=int(y_pred_reg)

# CSS pour centrer, agrandir le texte et ajouter un fond coloré
 st.markdown(
    f"""
    <style>
    .centered {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 10vh;
        font-size: 50px;
        font-weight: bold;
        background-color: lightblue; 
        color: black;  
        border-radius: 10px;
        padding: 20px;  /* pour ajouter du padding autour du texte */
    }}
    </style>
    <div class="centered">{resultat}</div>
    """,
    unsafe_allow_html=True
 )
    



