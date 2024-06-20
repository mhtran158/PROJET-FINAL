import streamlit as st

st.subheader('Modelisation : Regression')


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib_inline
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import sklearn.metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score




def Modelisation_Regression():
 st.title("Modèles de Regression")   
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
 
 st.write("<h2 style='color: blue; font-size: 20px;'>Notre dataset après nettoyage:</h2>", unsafe_allow_html=True)
 st.dataframe(data_target13.head())
  
 feats = data_target13.drop('CO2', axis = 1)
 target = data_target13['CO2'] 
  
 X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.20, random_state = 42)
  
 num = ['Poids (kg)', 'Cylindrée (cm3)', 'Puissance (ch)', 'Dimension (mm)', 'Autonomie electrique (km)']
 num_train = X_train[num]
 num_test = X_test[num]

 cat = ['fuel_mode','fuel_type'] 
 cat_train = X_train[cat]
 cat_test = X_test[cat]
  
 encoder = OneHotEncoder()
  
 one_cat_train_array = encoder.fit_transform(cat_train).toarray() 
 one_cat_train_df = pd.DataFrame(one_cat_train_array, columns=encoder.get_feature_names_out())
 one_cat_train_df.head()
  
 one_cat_test_array = encoder.transform(cat_test).toarray() 
 one_cat_test_df = pd.DataFrame(one_cat_test_array, columns=encoder.get_feature_names_out())
 one_cat_test_df.head(10)
  

 sc = StandardScaler()
 num_train_scaled_array = sc.fit_transform(num_train)
 num_train_scaled_df = pd.DataFrame(num_train_scaled_array, columns=sc.get_feature_names_out())
 num_test_scaled_array = sc.transform(num_test)
 num_test_scaled_df = pd.DataFrame(num_test_scaled_array, columns=sc.get_feature_names_out())
 
 X_train_new = pd.concat([num_train_scaled_df, one_cat_train_df.set_index(num_train_scaled_df.index)], axis=1)
 X_test_new = pd.concat([num_test_scaled_df, one_cat_test_df.set_index(num_test_scaled_df.index)], axis=1)
 X_test_new.head(10)
  
 regressor_lin=joblib.load("model_reg")
 regressor_random_forest=joblib.load('model_rf')
 regressor_decision_tree=joblib.load("model_dc")
  
   
 model_choisi=st.selectbox(label="Modèle",options=['Régression Linéaire', 'Decision Tree','Random Forest'])
 
 
 def train_model(model_choisi):
     if model_choisi=='Régression Linéaire':
         model=regressor_lin
     elif model_choisi=='Random Forest':
         model=regressor_random_forest
     elif model_choisi=='Decision Tree':
         model=regressor_decision_tree
     model.fit(X_train_new, y_train)
     return model
     
 md = ['Régression Linéaire', 'Random Forest', 'Decision Tree']
 choix=st.selectbox('Coefficient de détermination',md)
 st.write('Modèle choisi:', choix)
 
 model=train_model(choix)
  

 st.write("Score du Train:")
 Train=model.score(X_train_new, y_train)
 st.write(Train)
  
 st.write("Score du Test:")
 Test=model.score(X_test_new, y_test)
 st.write(Test)
  
  
 score = {'Score' :['Score train RL', 'Score test RL','Score train DC','Score test DC','Score train RF','Score test RF'],
           'Valeur': [regressor_lin.score(X_train_new,y_train),
                      regressor_lin.score(X_test_new,y_test),
                      regressor_decision_tree.score(X_train_new,y_train),
                      regressor_decision_tree.score(X_test_new,y_test),
                      regressor_random_forest.score(X_train_new,y_train),
                      regressor_random_forest.score(X_test_new,y_test)
                      ]}

 score = pd.DataFrame(score)
 score['Formatted_Value'] = score['Valeur'].apply(lambda x: '{:.2%}'.format(x)).str.replace('.', ',')
 score.to_csv('score.csv', index=False)
 score = pd.read_csv('score.csv',sep=',', index_col=False)
 st.write("<h2 style='color: blue; font-size: 20px;'>Tableau Score de différents modèles:</h2>", unsafe_allow_html=True)
  
 st.dataframe(score.head(6)) 
  
 
  
 st.write("<h2 style='color: blue; font-size: 20px;'>Graphique de score:</h2>", unsafe_allow_html=True)
 fig = px.bar(score, x='Score', y='Formatted_Value', text="Formatted_Value")
 fig.update_traces(texttemplate='%{text}', textposition='outside')
 st.plotly_chart(fig, use_container_width=True)
 pred_test = regressor_lin.predict(X_test_new)
 plt.scatter(pred_test, y_test, c='green')
 st.write("<h2 style='color: blue; font-size: 20px;'>Droite de Régression Linéaire:</h2>", unsafe_allow_html=True)
 plt.figure(figsize=(10, 10))
  
 plt.scatter(pred_test,y_test, color='green', label='Droite de régression RL')
 plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
 plt.xlabel("prediction")
 plt.ylabel("vrai valeur")
 st.set_option('deprecation.showPyplotGlobalUse', False)
 st.pyplot()


  

#Features importances:
 st.write("<h2 style='color: blue; font-size: 20px;'>Graphique de Features Importances de modèle Décision Tree:</h2>", unsafe_allow_html=True)  
 feat_importances = pd.DataFrame(regressor_decision_tree.feature_importances_, index=X_train_new.columns, columns=["Importance"])
 feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
 feat_importances.plot(kind='bar', figsize=(8,6))
 st.set_option('deprecation.showPyplotGlobalUse', False)
 st.pyplot()    
  
#Features importances:
 st.write("<h2 style='color: blue; font-size: 20px;'>Graphique de Features Importances de modèle Random Forest:</h2>", unsafe_allow_html=True)  
 feat_importances = pd.DataFrame(regressor_random_forest.feature_importances_, index=X_train_new.columns, columns=["Importance"])
 feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
 feat_importances.plot(kind='bar', figsize=(8,6))
 st.set_option('deprecation.showPyplotGlobalUse', False)
 st.pyplot()    
  
#Résidus:
 y_pred_decision_tree = regressor_decision_tree.predict(X_test_new)
 y_pred_train_decision_tree = regressor_decision_tree.predict(X_train_new)

 mae_decision_tree_train = mean_absolute_error(y_train,y_pred_train_decision_tree)
 mse_decision_tree_train = mean_squared_error(y_train,y_pred_train_decision_tree,squared=True)
 rmse_decision_tree_train = mean_squared_error(y_train,y_pred_train_decision_tree,squared=False)

 mae_decision_tree_test = mean_absolute_error(y_test,y_pred_decision_tree)
 mse_decision_tree_test = mean_squared_error(y_test,y_pred_decision_tree,squared=True)
 rmse_decision_tree_test = mean_squared_error(y_test,y_pred_decision_tree,squared=False)

     
 regressor_random_forest.fit(X_train_new, y_train)
 y_pred_random_forest = regressor_random_forest.predict(X_test_new)
 y_pred_random_forest_train = regressor_random_forest.predict(X_train_new)

 mae_random_forest_train = mean_absolute_error(y_train,y_pred_random_forest_train)
 mse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=True)
 rmse_random_forest_train = mean_squared_error(y_train,y_pred_random_forest_train,squared=False)

 mae_random_forest_test = mean_absolute_error(y_test,y_pred_random_forest)
 mse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=True)
 rmse_random_forest_test = mean_squared_error(y_test,y_pred_random_forest,squared=False)

 data = {'MAE train': [mae_decision_tree_train, mae_random_forest_train],
          'MAE test': [mae_decision_tree_test, mae_random_forest_test],
          'MSE train': [mse_decision_tree_train,mse_random_forest_train],
          'MSE test': [mse_decision_tree_test,mse_random_forest_test],
          'RMSE train': [rmse_decision_tree_train, rmse_random_forest_train],
          'RMSE test': [rmse_decision_tree_test, rmse_random_forest_test]}

 df = pd.DataFrame(data, index = ['Decision Tree', 'Random Forest '])
  
 df.to_csv('MAE.csv')
 MAE = pd.read_csv('MAE.csv',sep=',')
 st.write("<h2 style='color: blue; font-size: 20px;'>Tableau de Résidus:</h2>", unsafe_allow_html=True) 
 st.dataframe(MAE.head()) 

 st.write("<h2 style='color: blue; font-size: 20px;'>Graphique de résidus:</h2>", unsafe_allow_html=True)
 fig=px.bar(MAE, x=['Decision Tree', 'Random Forest'], y= ['MAE train', 'MAE test','MSE train','MSE test', 'RMSE train', 'RMSE test'])
 st.plotly_chart(fig, use_container_width=True)
  
  
 result=pd.DataFrame()
 y_pred=y_pred_random_forest
 result['y_test']=y_test
 result['y_pred']=y_pred
 result['residus']=result.y_test - result.y_pred
 quantiles=result.residus.quantile([0.1,0.25,0.75,0.9])
 st.write("<h2 style='color: blue; font-size: 20px;'>Quantiles des résidus:</h2>", unsafe_allow_html=True)
 st.write (quantiles)
  

 st.write("<h2 style='color: blue; font-size: 20px;'>Graphique de résidus de Random Forest:</h2>", unsafe_allow_html=True)
 sns.relplot(data=result,x='y_test', y='residus',alpha=0.5, height=8, aspect=10/8)
 plt.plot([0,result.y_test.max()],[0,0], 'r-.,')
 plt.plot([0,result.y_test.max()],[quantiles[0.10],quantiles[0.10]],'y--',label="80% des résidus présents dans cet intervalle")
 plt.plot([0,result.y_test.max()],[quantiles[0.90],quantiles[0.90]],'y--')
 plt.plot([0,result.y_test.max()],[quantiles[0.25],quantiles[0.25]],'y--',label="50% des résidus présents dans cet intervalle")
 plt.plot([0,result.y_test.max()],[quantiles[0.75],quantiles[0.75]],'y--')
 plt.xlim(0,result.y_test.max()+10)
 plt.xlabel('y_test')
 plt.ylabel('test résidus')
 plt.title('Résidus')
 plt.legend()
 st.pyplot()
  
  #Résidus en valeur absolue:
  
 residus=abs(y_test - y_pred_random_forest)
 residus.name='Résidus abs'
 sns.histplot(residus)
 st.pyplot()   
  
  #Modèle benmark dit "naif":
      
 y_pred=np.ones(len(y_test))*y_train.mean()
 st.write("<h2 style='color: blue; font-size: 20px;'>Modèle Naif:</h2>", unsafe_allow_html=True)
 st.write('r2:',format(round(r2_score(y_test,y_pred),2)))
 st.write('MAE naif:',format(round(mean_absolute_error(y_test,y_pred),2)))
 st.write('RMSE naif:',format(round(np.sqrt(mean_squared_error(y_test,y_pred)),2)))    
  
  

  