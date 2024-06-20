# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 06:34:49 2024

@author: chau
"""

import streamlit as st
from Home import Home
from Contexte import Contexte
from Donnees import Donnees
from Modelisation_Classification import Modelisation_Classification
from Modelisation_Regression import Modelisation_Regression
from Demo import Demo
from Limites_et_conclusion import Limites_et_conclusion

def main():
    st.sidebar.title("Navigation")
    options = ["Home", "Contexte", "Donnees", "Modelisation_Classification", "Modelisation_Regression", "Demo", "Limites_et_conclusion"]
    choix = st.sidebar.radio("Sélectionnez une page", options)
    
    if choix == "Home":
        Home()
    elif choix == "Contexte":
        Contexte()
    elif choix == "Donnees":
        Donnees()
    elif choix == "Modelisation_Classification":
        Modelisation_Classification()
    elif choix == "Modelisation_Regression":
        Modelisation_Regression()
    elif choix == "Demo":
        Demo()
    elif choix == "Limites_et_conclusion":
        Limites_et_conclusion()

if __name__ == "__main__":
    main()