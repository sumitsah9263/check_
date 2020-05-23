import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from xgboost import XGBClassifier
import pickle

df = pd.read_csv('creditcard.csv')

df_t = df['Class']
df_f = df.drop('Class',axis = 1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_f.values)
df_f_scaled = pd.DataFrame(scaled_features, columns= df_f.columns)

df_d = df.drop(['V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18'],axis = 1)
df_f = df.drop(df_d,axis=1)
df_t = df['Class']

clf1 = XGBClassifier(random_state = 42)

# Implementing Undersampling for Handling Imbalanced
nm = NearMiss(sampling_strategy = {0:100000 , 1: 492})
X_res,y_res=nm.fit_sample(df_f,df_t)

df3_train,df3_test,df_t_train,df_t_test = train_test_split(X_res,y_res,test_size = .3,stratify = y_res,random_state = 42)

clf1.fit(df3_train,df_t_train)

# Saving model to disk
pickle.dump(clf1, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(df3_test))

