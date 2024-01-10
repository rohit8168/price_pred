#!/usr/bin/env python
# coding: utf-8



# In[2]:

import sklearn


import pandas as pd 

import numpy as np


# In[20]:


df =pd.read_csv(r"laptopPrice.csv")
df.drop(['Number of Ratings','Number of Reviews'],axis=1,inplace=True)


# In[22]:


df.isna().sum()
print()


# In[23]:


rep={'acer':'other',"MSI":"other","APPLE":"other","Avita":"other"}
df['brand'].replace(rep,inplace=True)


# In[24]:


df['ram_gb']=df['ram_gb'].str.replace("GB","")
df['ram_gb']=df['ram_gb'].astype(int)
df['ssd']=df['ssd'].str.replace("GB","")
df['ssd']=df['ssd'].astype(int)
df['graphic_card_gb']=df['graphic_card_gb'].str.replace("GB","")
df['graphic_card_gb']=df['graphic_card_gb'].astype(int)
df['warranty']=df['warranty'].str.replace("year","").replace("years","")
df['warranty']=df['warranty'].str.replace("No warranty","0")
df['hdd']=df['hdd'].str.replace("GB","")
df['hdd']=df['hdd'].astype(int)
df['rating']=df['rating'].str.replace("stars","")
df['rating']=df['rating'].replace("1 star","1")
df['rating']=df['rating'].astype(int)
# Replace non-numeric characters in 'column_name' with an empty string
df['warranty'] = df['warranty'].str.replace(r'\D', '', regex=True)

df['warranty']=df['warranty'].astype(int)


# In[25]:


X=df.copy()


# In[26]:


x=df.value_counts(df['brand'])



# In[9]:


from sklearn.preprocessing import LabelEncoder
encoding=LabelEncoder()
df["processor_brand"]=encoding.fit_transform(df["processor_brand"])
df["processor_gnrtn"]=encoding.fit_transform(df["processor_gnrtn"])
df["ram_type"]=encoding.fit_transform(df["ram_type"])
df["os"]=encoding.fit_transform(df["os"])
df["weight"]=encoding.fit_transform(df["weight"])
df["Touchscreen"]=encoding.fit_transform(df["Touchscreen"])
df["brand"]=encoding.fit_transform(df["brand"])
df["processor_name"]=encoding.fit_transform(df['processor_name'])
df['os_bit']=encoding.fit_transform(df['os_bit'])
df['msoffice']=encoding.fit_transform(df['msoffice'])


# In[10]:




# In[11]:


y=df['Price']
df.drop("Price",inplace=True,axis=1)


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.2)


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[14]:


scaled_x_train=scaler.fit_transform(x_train.select_dtypes(include=["float","int32","int64"]))
scaled_x_test=scaler.fit_transform(x_test.select_dtypes(include=["float","int32","int64"]))
x=x_test.iloc[1]


# In[15]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(scaled_x_train,y_train)


# In[16]:


pred=regressor.predict(x_test)


# In[17]:


from sklearn.metrics import mean_squared_error,accuracy_score
np.sqrt(mean_squared_error(y_test,pred))


# In[18]:





# In[19]:





# In[28]:


import streamlit as st


# In[ ]:


st.title("Welcome,Laptop price pridicton")
brand_op=['Asus','DELL','LEANVO','HP','OTHER']
brand_op_map={
    'Asus':0,'DELL':1,'LEANVO':3,'HP':2,'OTHER':4
}
brand=st.selectbox("Brand",options=brand_op)
if brand in brand_op_map:
    brand=brand_op_map[brand]
processor_brand=['INTEL',"AMD","M1"]
processor_bra=st.selectbox("Processor Brand",options=processor_brand)
if processor_brand=="INTEL":
    processor_brand=1
elif processor_brand=="AMD":
    processor_brand=0
else:
    processor_brand=2
processor_name=["Core i5","Core i3","Core i7","Ryzen5","Ryzen7","Ryzen9","Ryzen3","celeron Dual","M1","corei9","pentinum quad"]
processor_map = {
    "Core i5": 2, "Core i3": 1, "Core i7": 3, "Ryzen5": 8, "Ryzen7": 9,
    "Ryzen9": 10, "Ryzen3": 7, "Celeron Dual": 0, "M1": 5, "Corei9": 4,
    "Pentium Quad": 6
}
process=st.selectbox("Processor name",options=processor_name)
if process in processor_map:
    process = processor_map[process]
process_gen=["12th","11th","10th","9th","8th","7th","4th","NOt Available"]
process_gen_map={
    "12th":2,"11th":1,"10th":0,"9th":6,"8th":5,"7th":4,"4th":3,"NOt Available":7
}
gen=st.selectbox("Gen",options=process_gen)
if gen in process_gen_map:
    process_gen=process_gen_map[gen]
ram_siz=[4,8,12,16]
ram_size=st.selectbox("RAM size",options=ram_siz)
ram_type=['DDR4','LPDDR4X','LPDDR4','LPDDR3','DDR5','DDR3']
ram_typ_map={
    'DDR4':1,'LPDDR4X':5,'LPDDR4':4,'LPDDR3':3,'DDR5':2,'DDR3':0
}
ram=st.selectbox("RAM Type",options=ram_type)
if ram in ram_typ_map:
    ram_type=ram_typ_map[ram]
ssd=[0,128,512,1024,2048,3072]
ssd_size=st.selectbox("SSD",options=ssd)
hdd=[0,256,512,1024,2048]
hdd=st.selectbox("HDD",options=hdd)
os=['Window','Mac','DOS']
os_map={
    'Window':2,'Mac':1,'DOS':0
}
os_=st.selectbox("OS",options=os)
if os_ in os_map:
    os_=os_map[os_]
os_bit=[32,64]
os_bit_=st.selectbox("os_bit",options=os_bit)
if os_bit==64:
    os_bit=1
else:
    os_bit=0
graphic=[0,2,4,6,8,16]
gra_siz=st.selectbox("graphic card",options=graphic)
weight=["Casual","ThinLight","Gaming"]
weight_map={
    "Casual":0,"ThinLight":2,"Gaming":1
}
weight=st.selectbox("weight",options=weight)
if weight in weight_map:
    weight=weight_map[weight]
warr=[0,1,2,3]
warr_=st.selectbox("warranty",options=warr)
Touchscreen=["NO","YES"]
touch=st.selectbox("Touchscreen",options=Touchscreen)
if touch=="NO":
    touch=0
else:
    touch=1
ms=["NO","YES"]
ms_o=st.selectbox("Ms Office",options=ms)
if ms_o=="NO":
    ms_o=0
else:
    ms_o=1
rate=[1,2,3,4,5]
rate_=st.selectbox("Rating",options=rate)
x=np.array([brand,processor_brand,process,process_gen,ram_size,ram_type,ssd_size,hdd,os_,os_bit,gra_siz,weight,warr_,touch,ms_o,rate_]).reshape(1,-1)
st.write(regressor.predict(x))
