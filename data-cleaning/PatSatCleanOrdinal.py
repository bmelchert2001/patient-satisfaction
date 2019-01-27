# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:36:14 2019

@author: melcb01
"""

import os
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# Set working directory to file source path
path = 'C:\\Users\melcb01\\.spyder-py3\\patsat'
os.chdir(path)

# Read Patient Satisfaction Excel file into DataFrame
#df = pd.read_excel('Patient_Satisfaction.xlsx')
df = pd.read_pickle('Patient_Satisfaction')

# Create Datetime index column from refDate duplicate (not part of actual DataFrame)
df['date_index'] = df.refDate
df.set_index('date_index', inplace=True)

# Convert object-type columns to categorical w/ for-loop
for col in ['p0003000','p0011900','p0013900','p0001500','p0000600','p0033200',
            'p0002100','surveyId','facilityId','surveyRespondentId']:
    df[col] = df[col].astype('category')

# First drop columns that Ko said were Not Related
df.drop(['p1400123','p0700403','p0401203','p0407602','p0110402','p0401147',
         'p0402702','p0600602','p0502502','p0117302','p0516302'],
        axis=1, inplace=True)

# Drop duplicate colums, columns with mainly same values, adding no variation
df.drop(['Nursing', 'Doctor', 'Room', 'Staff', 'vendorName', 'vendorId', 
         'surveyType', 'facilityName', 'surveyRespondentId', 'p0002100', 
         'totalexpense', 'payrollexpense', 'personnel', 'TotalperPersonnel', 
         'Payrollperpersonnel', 'BedperPersonnel', 'AdmperPersonnel'], 
        axis=1, inplace=True)

# Drop Null rows from Related columns w/ for-loop
for col in ['p1400120', 'p1400220', 'p1400320', 'p1400420', 'p1400520', 
            'p1400620', 'p1400720', 'p1400820', 'p1400920', 'p1400403', 
            'p1400503', 'p1401020', 'p1401120', 'p1401220', 'p1401320', 
            'p1401420', 'p0200202', 'p0801402', 'p0302902', 'p1102102', 
            'p0340602', 'p0349502', 'p0501102', 'p0500302', 'p0001100', 
            'p0001200', 'p1400102', 'p1500526', 'p0700102', 'p1400143', 
            'p1400243', 'p1400343', 'p1400202', 'p0610503', 'p0610603', 
            'p0610703', 'p0610803', 'p0610903', 'p1400203', 'p1400103']:
    df.dropna(subset = [col], inplace=True)

# Drop columns with majority Null values
df.drop(['p0341602', 'p0370302', 'p0604099', 'p0411002', 'p0401339', 'p0607002', 
         'p0602302', 'p0001100', 'p0033200', 'p0000600', 'p0001500', 'p0011900', 
         'p0003000'], axis=1, inplace=True)

# Drop Null rows from Maybe Related Columns
for col in ['p0302402', 'p1400303']:
    df.dropna(subset = [col], inplace=True)

# Drop any rows with Null values still remaining
df = df.dropna()

# Convert datetime column to ordinal 
df['refDate'] = df['refDate'].apply(lambda x: x.toordinal())

# Column names stored for later reference
col_names = list(df.columns.values)
X_names = list(df.iloc[:,3:].columns.values)
y_names = list(df.iloc[:,:3].columns.values)

# Encode categorical(ordinal) features (encode X variables or not)
from sklearn.preprocessing import LabelEncoder
Xv = df.iloc[:,3:].values
enc = LabelEncoder()
Xc = df.iloc[:,3:]
Xc = Xc.apply(enc.fit_transform)

# Standardize, normalize the features for activation functions
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
Xs = StandardScaler().fit_transform(Xv)
Xn5 = MinMaxScaler(feature_range=(-.5,.5)).fit_transform(Xs)
Xn = MinMaxScaler(feature_range=(0,1)).fit_transform(Xs)

# Target variables 
y14 = df.iloc[:,0]
y05 = df.iloc[:,1]
yH = df.iloc[:,2]

# Transform Target variables into categorical for classification
bins1 = (-1, 9.0, 10.0)  # -1, 4.0, 7.0, 9.0, 10.0
group_names1 = ['Low', 'Perfect']  # Low, Mid, High, Perfect
howcat1 = pd.cut(yH, bins1, labels=group_names1)
yH2 = enc.fit(howcat1)
yH2 = enc.transform(howcat1)
# Create standardized and normalized Labels
yH2s = StandardScaler().fit_transform(Xv)
yH25 = MinMaxScaler(feature_range=(-.5,.5)).fit_transform(Xs)
yH2n = MinMaxScaler(feature_range=(0,1)).fit_transform(Xs)

bins2 = (-1, 4.0, 9.0, 10.0)  # -1, 4.0, 7.0, 9.0, 10.0
group_names2 = ['Low', 'Mid', 'Perfect']  # Low, Mid, High, Perfect
howcat2 = pd.cut(yH, bins2, labels=group_names2)
yH3 = enc.fit(howcat2)
yH3 = enc.transform(howcat2)
yH3s = StandardScaler().fit_transform(Xv)
yH35 = MinMaxScaler(feature_range=(-.5,.5)).fit_transform(Xs)
yH3n = MinMaxScaler(feature_range=(0,1)).fit_transform(Xs)

bins3 = (-1, 4.0, 7.0, 9.0, 10.0)  # -1, 4.0, 7.0, 9.0, 10.0
group_names3 = ['Low', 'Mid', 'High', 'Perfect']  # Low, Mid, High, Perfect
howcat3 = pd.cut(yH, bins3, labels=group_names3)
yH4 = enc.fit(howcat3)
yH4 = enc.transform(howcat3)
yH4s = StandardScaler().fit_transform(Xv)
yH45 = MinMaxScaler(feature_range=(-.5,.5)).fit_transform(Xs)
yH4n = MinMaxScaler(feature_range=(0,1)).fit_transform(Xs)

bins4 = (-1, 3.0, 4.0)
group_names4 = ['Imperfect', 'Perfect']
howcat4 = pd.cut(y14, bins4, labels=group_names4)
y14a = enc.fit(howcat4)
y14a = enc.transform(howcat4)
y14as = StandardScaler().fit_transform(Xv)
y14a5 = MinMaxScaler(feature_range=(-.5,.5)).fit_transform(Xs)
y14an = MinMaxScaler(feature_range=(0,1)).fit_transform(Xs)

bins5 = (-1, 1.0, 3.0, 4.0)
group_names5 = ['Low', 'Mid', 'Perfect']
howcat5 = pd.cut(y14, bins5, labels=group_names5)
y14b = enc.fit(howcat5)
y14b = enc.transform(howcat5)
y14bs = StandardScaler().fit_transform(Xv)
y14b5 = MinMaxScaler(feature_range=(-.5,.5)).fit_transform(Xs)
y14bn = MinMaxScaler(feature_range=(0,1)).fit_transform(Xs)

bins6 = (-1, 4.0, 5.0)
group_names6 = ['Imperfect', 'Perfect']
howcat6 = pd.cut(y05, bins6, labels=group_names6)
y05a = enc.fit(howcat6)
y05a = enc.transform(howcat6)
y05as = StandardScaler().fit_transform(Xv)
y05a5 = MinMaxScaler(feature_range=(-.5,.5)).fit_transform(Xs)
y05an = MinMaxScaler(feature_range=(0,1)).fit_transform(Xs)

bins7 = (-1, 1.0, 3.0, 5.0)
group_names7 = ['Low', 'Mid', 'Perfect']
howcat7 = pd.cut(y05, bins7, labels=group_names7)
y05b = enc.fit(howcat7)
y05b = enc.transform(howcat7)
y05bs = StandardScaler().fit_transform(Xv)
y05b5 = MinMaxScaler(feature_range=(-.5,.5)).fit_transform(Xs)
y05bn = MinMaxScaler(feature_range=(0,1)).fit_transform(Xs)