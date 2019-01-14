# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:08:05 2018

@author: melcb01
"""

import os
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# Set working directory to file source path
path = 'c:\\users\\melcb01\\.spyder-py3\\patsat'
os.chdir(path)

# Read Patient Satisfaction Excel file into DataFrame
#df = pd.read_excel('Patient_Satisfaction.xlsx')
df = pd.read_pickle('Patient_Satisfaction')

# Create Datetime index column from refDate duplicate (not part of actual DataFrame)
df['date_index'] = df.refDate
df.set_index('date_index', inplace=True)

# Convert object-type columns to categorical w/ for-loop
for col in ['p0003000','p0011900','p0013900','p0001500','p0000600','p0033200','p0002100','surveyId','facilityId','surveyRespondentId']:
    df[col] = df[col].astype('category')

# First drop columns that Ko said were Not Related
df.drop(['p1400123','p0700403','p0401203','p0407602','p0110402','p0401147','p0402702','p0600602','p0502502','p0117302','p0516302'], axis=1, inplace=True)

## Drop Target columns not being used for this model
#df.drop(['p1400122', 'p0525402', 'howlikly'], axis=1, inplace=True)

# Drop duplicate colums, columns with mainly same values, adding no variation
df.drop(['vendorName', 'vendorId', 'surveyType', 'facilityName', 'surveyRespondentId', 'p0002100', 'totalexpense', 'payrollexpense', 'personnel', 'TotalperPersonnel', 'Payrollperpersonnel', 'BedperPersonnel', 'AdmperPersonnel'], axis=1, inplace=True)

# Drop Null rows from Related columns w/ for-loop
for col in ['p1400120', 'p1400220', 'p1400320', 'p1400420', 'p1400520', 'p1400620', 'p1400720', 'p1400820', 'p1400920', 'p1400403', 'p1400503', 'p1401020', 'p1401120', 'p1401220', 'p1401320', 'p1401420', 'p0200202', 'p0801402', 'p0302902', 'p1102102', 'p0340602', 'p0349502', 'p0501102', 'p0500302', 'p0001100', 'p0001200', 'p1400102', 'p1500526', 'p0700102', 'p1400143', 'p1400243', 'p1400343', 'p1400202', 'p0610503', 'p0610603', 'p0610703', 'p0610803', 'p0610903', 'p1400203', 'p1400103']:
    df.dropna(subset = [col], inplace=True)
    #print(col, df.shape)

#print(df.info())

# Drop columns with majority Null values
df.drop(['p0341602', 'p0370302', 'p0604099', 'p0411002', 'p0401339', 'p0607002', 'p0602302', 'p0001100', 'p0033200', 'p0000600', 'p0001500', 'p0011900', 'p0003000'], axis=1, inplace=True)

# Drop Null rows from Maybe Related Columns
for col in ['p0302402', 'p1400303']:
    df.dropna(subset = [col], inplace=True)
    #print(col, df.shape)

# Drop rows with Null values remaining
df = df.dropna()

# Convert datetime column to ordinal 
df['refDate'] = df['refDate'].apply(lambda x: x.toordinal())

# Column names for reference
col_names = list(df.columns.values)
X_names = list(df.iloc[:,3:].columns.values)
y_names = list(df.iloc[:,:3].columns.values)

df2 = df.shape

# Encode categorical features into numerical format 
cat = ['sdmPk','facilityId','surveyId','refDate','p1400120','p1400220','p1400320','p1400420','p1400520','p1400620','p1400720','p1400820','p1400920','p1400403','p1400503','p1401020','p1401120','p1401220','p1401320','p1401420','p0200202','p0801402','p0302402','p0302902','p1102102','p0340602','p0349502','p0501102','p0500302','p0001200','p1400102','p1500526','p0700102','p1400143','p1400243','p1400343','p0003100','p0300502','p0013900','p1400202','p0610503','p0610603','p0610703','p0610803','p0610903','p1400203','p1400103','p1400303','Age1','Nursing','Doctor','Room','Staff','Black','Asian','Pacific','Indian','White','staffedbeds','admissions','census','outpatientvisits','births']

for col in cat:
    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
    df.drop([col], axis=1, inplace=True)

# Standardize, normalize the features for PCA
X = df.iloc[:,3:].values
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
Xs = StandardScaler().fit_transform(X)
Xn = MinMaxScaler().fit_transform(Xs)

# Perform PCA to identify primary columns
from sklearn.decomposition import PCA
pca = PCA(n_components=80)
Xp = pca.fit_transform(Xn)

# Target variables
y14 = df.iloc[:,0]
y05 = df.iloc[:,1]
yH = df.iloc[:,2]

# Transform Target column into categorical for classification
from sklearn.preprocessing import LabelEncoder
bins = (-1, 4.0, 7.0, 9.0, 10.0)
group_names = ['Low', 'Mid', 'High', 'Perfect']
howcats = pd.cut(yH, bins, labels=group_names)
enc = LabelEncoder()
yH = enc.fit(howcats)
yH = enc.transform(howcats)

bins = (-1, 1.0, 3.0, 4.0)
group_names = ['Low', 'Mid', 'Perfect']
howcat = pd.cut(y14, bins, labels=group_names)
y05 = enc.fit(howcat)
y05 = enc.transform(howcat)

bins = (-1, 1.0, 3.0, 5.0)
group_names = ['Low', 'Mid', 'Perfect']
howcat = pd.cut(y05, bins, labels=group_names)
y05 = enc.fit(howcat)
y05 = enc.transform(howcat)

# File save info
df.to_pickle('StandardCats')

#writer = pd.ExcelWriter('patsatanalysis.xlsx')
#df.to_excel(writer,'Sheet1')
#writer.save()