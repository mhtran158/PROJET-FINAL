import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def Modelisation_Classification():
    st.title("Modèles de Classification")

    data_target13C = pd.read_csv('data_target13_New.csv')
    #st.sidebar.title("Sommaire")
    pages=["Modélisation Initiale", "Optimisations"]
    page=st.sidebar.radio("Aller vers", pages)

    if page == pages[0] :
        st.subheader("Modelisation Initiale")
        if st.checkbox("Dataset après nettoyage") :
            st.dataframe(data_target13C.head())
            st.write("Dimensions :")
            st.write(data_target13C.shape)

        feats = data_target13C.drop(['CO2'], axis = 1)
        target = data_target13C[['CO2']]

        X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.20, random_state = 42)

        num_train = X_train[['masse', 'cylindrée', 'puissance', 'dimension', 'autonomie électrique']]
        num_test = X_test[['masse', 'cylindrée', 'puissance', 'dimension', 'autonomie électrique']]
        cat_train = X_train[['fuel mode','fuel type']]
        cat_test = X_test[['fuel mode','fuel type']]

        encoder = OneHotEncoder() # creation de 'encoder' un objet OneHotEncoder
        one_cat_train_array = encoder.fit_transform(cat_train).toarray() 
        one_cat_train_df = pd.DataFrame(one_cat_train_array, columns=encoder.get_feature_names_out())
        one_cat_test_array = encoder.transform(cat_test).toarray() 
        one_cat_test_df = pd.DataFrame(one_cat_test_array, columns=encoder.get_feature_names_out())
    
        le = LabelEncoder()
        label_train_array = le.fit_transform(y_train)
        label_train_df = pd.DataFrame(label_train_array)
        #st.dataframe(label_train_df())
        label_test_array = le.fit_transform(y_test)
        label_test_df = pd.DataFrame(label_test_array)
        #st.dataframe(label_test_df())
                
        sc = StandardScaler()
        num_train_scaled_array = sc.fit_transform(num_train)
        num_train_scaled_df = pd.DataFrame(num_train_scaled_array, columns=sc.get_feature_names_out())
        num_test_scaled_array = sc.transform(num_test)
        num_test_scaled_df = pd.DataFrame(num_test_scaled_array, columns=sc.get_feature_names_out())

        X_train_new = pd.concat([num_train_scaled_df, one_cat_train_df.set_index(num_train_scaled_df.index)], axis=1)
        X_test_new = pd.concat([num_test_scaled_df, one_cat_test_df.set_index(num_test_scaled_df.index)], axis=1)

        if st.checkbox("Résultats de modelisation") :

            def prediction(modele):
                if modele == 'Regression Logistique':
                    clf = LogisticRegression()
                elif modele == 'Arbre de décision':
                    clf = DecisionTreeClassifier(random_state=50)
                elif modele == 'Random Forest':
                    clf = RandomForestClassifier()
                clf.fit(X_train_new, y_train)
                return clf


            choix = ['Arbre de décision', 'Random Forest', 'Regression Logistique']
            option = st.selectbox('Veuillez choisir un modèle', choix)
            st.write('Le modèle choisi est :', option)

            if st.button("Exécution", key = "classify"):

                clf = prediction(option)

                st.write("Accuracy Train Score :")
                Train = clf.score(X_train_new, y_train)
                st.write(Train)
                            
                st.write("Accuracy Test Score") 

                Test = clf.score(X_test_new, y_test)
                st.write(Test)

                ### Confusion Matrix
                st.write("Matrice de confusion")
                y_pred = clf.predict(X_test_new)

                cm = confusion_matrix(y_test, y_pred)

                ## Get Class Labels
                labels = le.classes_
                class_names = labels

                # Plot confusion matrix
                fig_cm, ax_cm = plt.subplots()
                fig_cm = plt.figure(figsize=(6,6))
                ax_cm= plt.subplot()
                sns.heatmap(cm, annot=True, ax = ax_cm, fmt='d', cmap='Blues')
                # labels, title and ticks
                ax_cm.set_xlabel('Prédictions', fontsize=12)
                ax_cm.xaxis.set_label_position('top')
                plt.xticks(rotation=90)
                ax_cm.xaxis.set_ticklabels(class_names, fontsize = 8)
                ax_cm.xaxis.tick_top()
                ax_cm.set_ylabel('Réalités', fontsize=12)
                ax_cm.yaxis.set_ticklabels(class_names, fontsize = 8)
                plt.yticks(rotation=0)
                #plt.title('Matrice de confusion', fontsize=12)
                plt.savefig('ConMat24.png')
                st.pyplot(fig_cm)

                st.write("Rapport de classification :")
                target_names = ["A", "B", "C", "D", "E", "F", "G"]
                Report = pd.DataFrame(classification_report(y_test, clf.predict(X_test_new),target_names=target_names, output_dict=True)).transpose()
                st.dataframe(Report)

                # Graphique des importances
                st.write("Les 5 premières variables importantes prises en compte dans la classification")

                feat_importances = pd.DataFrame(clf.feature_importances_, index=X_train_new.columns, columns=["Importance"])
                feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
                feat_importances.to_csv('feat_importances.csv', index=False)
                st.dataframe(feat_importances.head())
            
                #fig = px.bar(feat_importances, x = X_train_new.columns, y = 'Importance')
                fig = px.bar(feat_importances)
                fig.update_layout(title= "Graphe des importances", 
                            title_font_size=20, xaxis_title="Variables", 
                            xaxis_title_font_size= 18, 
                            yaxis_title="Taux", 
                            yaxis_title_font_size= 18)
                st.plotly_chart(fig)

                st.write(" Forte influence de la variable masse suivie de l’autonomie électrique, la dimension, la puissance et la cylindrée.")
                st.write(" Imprtance faible (inférieure à 5 %) pour les variables types de carburant pétrole, diesel ainsi que les véhicules de mode M(Mono-carburant)" 
                         " et H(Hybride non rechargeable sur borne électrique).")


            if st.checkbox("Analyses") :
                st.write("Du point de vue des scores accuracy qui indiquent pour un modèle le taux de prédictions correctes des classes sur le total des prédictions,"
                        " le modèle RandomForest rfcl1 obtient le meilleur score de 91,74% de prédictions correctes sur l’ensemble train et de 85,73 % sur l’ensemble test."
                        " Le modèle dectree1(Arbre de décision) obtient un score test moindre (85,34%) comparé à celui du RandomForest mais les scores train"
                        " pour les deux modèles restent identiques. Cependant on note un surapprentissage de l’ordre de 6% pour le modèle rfcl1 contre 6,41% pour le modèle dectree1."
                        " Malgré une absence de surapprentissage sur le modèle de régression logistique reglog1, ce dernier s’avère moins précis dans ses prédictions à la fois sur les ensembles train et"
                        " test (score train : 69,60% ; score test : 69,66%) en faisant une comparaison par rapport aux deux autres modèles.")
                st.write(" En analysant les matrices de confusion et les rapports de classification des 3 modèles, La classe B est mal classifiée par le modèle reglog1"
                        " en classifiant la majorité des véhicules catégorisés dans la réalité en classe B en classe C (714 véhicules). Avec un taux de précision de 66%, "
                        " le taux de rappel de 29 % est très faible par rapport à celui des autres classes. La classe A est cependant la mieux classée avec 98% de bonnes prédictions"
                        " et le meilleur taux de rappel de 96%. Sur 1341 véhicules, 1288 ont été correctement prédits.")
                st.write("L’arbre de décision dectree1 prédit mieux la classe B comparé au modèle précédent de régression logistique reglog1 avec 757 bonnes prédictions sur 1037"
                         " au total soit une précision 76% et taux de rappel de 73% contre 29% constaté auparavant. La classe B reste toujours la classe la moins bien prédite "
                         " et la classe A la mieux classifiée (1325 véhicules correctement prédits en classe A sur 1341 véhicules de classe A réels). Les classes C à G "
                         " sont correctement prédites à minima à 80% avec en tête la classe G catégorisée correctement avec une précision de 92% et un coefficient de rappel de 91%.")
                st.write("Par le modèle RandomForest rfcl1, on note une amélioration sur les taux de rappel pour les classes D, F et G. Ainsi , comparé à la réalité, les nombres"
                        " d’individus correctement prédits par ce modèle est meilleur par rapport à ceux obtenus précédemment. Le coefficient de rappel de la classes B"
                        " s'est dégradé au profit du taux de précision. . Les taux de rappel sont meilleurs pour les classes D, F et G." 
                        " De la classe A à la classe G, le modèle RandomForest rfcl1 se trompe moins en réalisant 86% de bonnes prédictions en moyenne.")
                st.write("Ainsi sur un jeu de test de 14074 véhicules :")
                st.write("•	Pour la classe A : 1320 de prédictions correctes sur un total de 1341 véhicules A")
                st.write("•	Pour la classe B : 742 de prédictions correctes sur un total de 1037 véhicules B")
                st.write("•	Pour la classe C : 2823 de prédictions correctes sur un total de 3303 véhicules C")
                st.write("•	Pour la classe D : 2309 de prédictions correctes sur un total de 2824 véhicules D")
                st.write("•	Pour la classe E : 2557 de prédictions correctes sur un total de 2980 véhicules E")
                st.write("•	Pour la classe F : 1448 de prédictions correctes sur un total de 1657 véhicules F")
                st.write("•	Pour la classe G : 867 de prédictions correctes sur un total de 932 véhicules G.")


    if page == pages[1] :
        st.subheader("Optimisations")
        if st.checkbox("Scores Acuracy train et test obtenus :"):
            df = pd.read_csv('Acuracy_scores.csv')
            st.dataframe(df)

        if st.checkbox("Grahiques des scores :"):
            df.sort_values(by='Score_test', ascending=False, inplace=True)

            fig = px.bar(df, x='Modele', y = ['Score_test', 'Score_train'], barmode= "group")

            fig.update_layout(title= "Accuracy Scores Train et Test des modèles entrainés", 
                        title_font_size=20, xaxis_title="Modèles", 
                        xaxis_title_font_size= 18, 
                        yaxis_title="Scores", 
                        yaxis_title_font_size= 18)
            st.plotly_chart(fig)

            if st.checkbox("Analyses") :
                st.write("Parmi les modèles testés dans le cadre de l’optimisation, les meilleurs scores test sont obtenus par les modèles de type randomforest(scores entre 85 et 86%)"
                         " notamment le rfcl3 qui comparé au modèle rfcl1 étudié plus haut améliore non seulement le score test (85,84% vs 85,73%) mais réduit aussi "
                         " le surapprentissage (5,89% vs 6%)")

                st.write("Les modèles de régressions logistiques affichent des scores train et test inférieurs à 70% mais n’ont pas de surapprentissage.") 
                st.write("Les scores des modèles SVM se situent entre 70% et 73% avec une particularité pour le modèle svm_poids entrainé selon le poids des différentes classes qui donne en revanche "
                        " un score train et score test quasi identiques de 73% avec un surapprentissage quasi nul.")
                st.write("Les modèles d’arbre de décision affichent des scores test au-dessus de 85% avec du surapprentissage supérieur à 6%.")
                st.write("Le modèle BalancedRandomForest bclf, avec un score train de 89% et test de 84,87% présente un surapprentissage de 4,21% qui reste moindre par rapport à celui des modèles "
                        " randomforest, arbre de décision et svm.")
                
                st.write("Zoom sur le modèle RandomForest rfcl3") 
                st.write("L’analyse de la matrice de confusion et du rapport de classification du modèle rfcl3 par rapport au modèle rfcl1 étudié plus haut permet de constater une amélioration"
                        " des coefficients de rappel des classes B, D et E augmentant ainsi les nombre de bonnes prédictions de ces trois classes. Les taux de précision et de rappel des autres" 
                        " classes restent néanmoins relativement stables.")
                st.write("Ainsi, par ce modèle rfcl3, nous obtenons comparé au rfcl1 :")
                st.write("•	Pour la classe A : 1318 vs 1320 de prédictions correctes pour le rfcl1,")
                st.write("•	Pour la classe B : 744 vs 742 de prédictions correctes pour le modèle rfcl1")
                st.write("•	Pour la classe C : 2823 vs 2823 de prédictions correctes pour le modèle rfcl1")
                st.write("•	Pour la classe D : 2313 vs 2309 de prédictions correctes pour le modèle rfcl1")
                st.write("•	Pour la classe E : 2572 vs 2557 de prédictions correctes pour le modèle rfcl1")
                st.write("•	Pour la classe F : 1445 vs 1448 de prédictions correctes pour le modèle rfcl1")
                st.write("•	Pour la classe G : 867 vs 867 de prédictions correctes pour le modèle rfcl1.")














