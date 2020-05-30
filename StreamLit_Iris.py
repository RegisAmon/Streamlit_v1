
# Importing Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Setting up the header 
st.title("Iris Dataset")
st.subheader("Complete Model Lifecycle")
# Providing a radio button to browse and upload the imput file 
filename = st.file_uploader("upload file", type = ("csv", "xlsx"))
data = pd.read_csv(filename)

#------------------------------------------------------------------------------
# To upload an input file from the specified path
#@st.cache(persist=True)
#def explore_data(dataset):
#    df = pd.read_csv(os.path.join(dataset))
#    return df
#data = explore_data(my_dataset)
#------------------------------------------------------------------------------

# Dataset preview
if st.checkbox("Preview Dataset"):
    if st.button("Head"):
        st.write(data.head())
    elif st.button("Tail"):
        st.write(data.tail())
    else:
        number = st.slider("Select No of Rows", 1, data.shape[0])
        st.write(data.head(number))


# show entire data
if st.checkbox("Show all data"):
    st.write(data)


# show column names
if st.checkbox("Show Column Names"):
    st.write(data.columns)

# show dimensions
if st.checkbox("Show Dimensions"):
    st.write(data.shape)
     
# show summary
if st.checkbox("Show Summary"):
    st.write(data.describe())
    
# show missing values
if st.checkbox("Show Missing Values"):
    st.write(data.isna().sum())    

# Select a column to treat missing values
col_option = st.selectbox("Select Column to treat missing values", data.columns) 

# Specify options to treat missing values
missing_values_clear = st.selectbox("Select Missing values treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))

if missing_values_clear == "Replace with Mean":
    replaced_value = data[col_option].mean()
    st.write("Mean value of column is :", replaced_value)
elif missing_values_clear == "Replace with Median":
    replaced_value = data[col_option].median()
    st.write("Median value of column is :", replaced_value)
elif missing_values_clear == "Replace with Mode":
    replaced_value = data[col_option].mode()
    st.write("Mode value of column is :", replaced_value)


Replace = st.selectbox("Replace values of column?", ("Yes", "No"))
if Replace == "Yes":
    data[col_option] = data[col_option].fillna(replaced_value)
    st.write("Null values replaced")
elif Replace == "No":
    st.write("No changes made")


# To change datatype of a column in a dataframe
# display datatypes of all columns
if st.checkbox("Show datatypes of the columns"):
    st.write(data.dtypes)

col_option_datatype = st.selectbox("Select Column to change datatype", data.columns) 

input_data_type = st.selectbox("Select Datatype of input column", (str,int, float))  
output_data_type = st.selectbox("Select Datatype of output column", (str,int, float))

st.write("Datatype of ",col_option_datatype," changed to ", output_data_type)

data[col_option_datatype] = output_data_type(data[col_option_datatype])


if st.checkbox("Show updated datatypes of the columns"):
    st.write(data.dtypes)


# visualization
# scatter plot
col1 = st.selectbox('Which feature on x?', data.columns)
col2 = st.selectbox('Which feature on y?', data.columns)
fig = px.scatter(data, x =col1,y=col2)
st.plotly_chart(fig)


# correlartion plots
if st.checkbox("Show Correlation plots with Seaborn"):
    st.write(sns.heatmap(data.corr()))
    st.pyplot()


# Machine Learning Algorithms
st.subheader('Machine Learning models')
 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
 
 
features = st.multiselect("Select Feature Columns",data.columns)
labels = st.multiselect("select target column",data.columns)

features= data[features].values
labels = data[labels].values


train_percent = st.slider("Select % to train model", 1, 100)
train_percent = train_percent/100

X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=train_percent, random_state=1)


alg = ['XGBoost Classifier', 'Support Vector Machine', 'Random Forest Classifier']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='XGBoost Classifier':
    XG = XGBClassifier()
    XG.fit(X_train, y_train)
    acc = XG.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_XG = XG.predict(X_test)
    cm_XG=confusion_matrix(y_test,pred_XG)
    st.write('Confusion matrix: ', cm_XG)
   
elif classifier == 'Support Vector Machine':
    svm=SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)

elif classifier == 'Random Forest Classifier':
    RFC=RandomForestClassifier()
    RFC.fit(X_train, y_train)
    acc = RFC.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_RFC = RFC.predict(X_test)
    cm=confusion_matrix(y_test,pred_RFC)
    st.write('Confusion matrix: ', cm)