#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:37:21 2017

@author: parth
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

train_df.drop(['Id','Utilities','KitchenQual', 'RoofMatl','Neighborhood', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','FireplaceQu','Electrical','KitchenAbvGr', 'EnclosedPorch','PavedDrive','LandContour','LandSlope', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)
test_df.drop(['Utilities','KitchenQual','FireplaceQu', 'Neighborhood','RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF','KitchenAbvGr','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Electrical','PavedDrive','LandContour','LandSlope', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)
#print(train_df.dtypes)
#print(train_df)
train_df['MSSubClass'] = train_df['MSSubClass'].astype(str)

# MSZoning NA in pred. filling with most popular values
train_df['MSZoning'] = train_df['MSZoning'].fillna(train_df['MSZoning'].mode()[0])

# LotFrontage  NA in all. I suppose NA means 0
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())

# Alley  NA in all. NA means no access
train_df['Alley'] = train_df['Alley'].fillna(0)

# Converting OverallCond to str
train_df.OverallCond = train_df.OverallCond.astype(str)

# MasVnrType NA in all. filling with most popular values
train_df['MasVnrType'] = train_df['MasVnrType'].fillna(train_df['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement


# TotalBsmtSF  NA in pred. I suppose NA means 0
train_df['TotalBsmtSF'] = train_df['TotalBsmtSF'].fillna(0)




# FireplaceQu  NA in all. NA means No Fireplace


# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage


# GarageCars  NA in pred. I suppose NA means 0
train_df['GarageCars'] = train_df['GarageCars'].fillna(0.0)

# SaleType NA in pred. filling with most popular values
train_df['SaleType'] = train_df['SaleType'].fillna(train_df['SaleType'].mode()[0])

# Year and Month to categorical
train_df['YrSold'] = train_df['YrSold'].astype(str)
train_df['MoSold'] = train_df['MoSold'].astype(str)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
train_df['TotalSF'] = train_df['TotalBsmtSF'] + train_df['1stFlrSF'] + train_df['2ndFlrSF']
train_df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)


test_df['MSSubClass'] = test_df['MSSubClass'].astype(int)

# MSZoning NA in pred. filling with most popular values
test_df['MSZoning'] = test_df['MSZoning'].fillna(test_df['MSZoning'].mode()[0])

# LotFrontage  NA in all. I suppose NA means 0
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())

# Alley  NA in all. NA means no access
test_df['Alley'] = test_df['Alley'].fillna(0)

# Converting OverallCond to str
test_df.OverallCond = test_df.OverallCond.astype(str)

# MasVnrType NA in all. filling with most popular values
test_df['MasVnrType'] = test_df['MasVnrType'].fillna(test_df['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement


# TotalBsmtSF  NA in pred. I suppose NA means 0
test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(0)

# Electrical NA in pred. filling with most popular values


# GarageCars  NA in pred. I suppose NA means 0
test_df['GarageCars'] = test_df['GarageCars'].fillna(0.0)

# SaleType NA in pred. filling with most popular values
test_df['SaleType'] = test_df['SaleType'].fillna(test_df['SaleType'].mode()[0])

# Year and Month to categorical
test_df['YrSold'] = test_df['YrSold'].astype(str)
test_df['MoSold'] = test_df['MoSold'].astype(str)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
test_df['TotalSF'] = test_df['TotalBsmtSF'] + test_df['1stFlrSF'] + test_df['2ndFlrSF']
test_df.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)##print(train_df['MSSubClass'])
#train_df=train_df[['MSSubClass','MSZoning','BldgType','OverallQual','OverallCond','SaleType','SaleCondition','SalePrice']]
##print(train_df)
#test_df=test_df[['Id','MSSubClass','MSZoning','BldgType','OverallQual','OverallCond','SaleType','SaleCondition']]

combine=[train_df,test_df]
for columns in test_df.columns:
    print(columns,test_df[columns].isnull().sum())
freq_port = train_df.BldgType.dropna().mode()[0]
#print(freq_port)
for dataset in combine:
    dataset['MSZoning'] = dataset['MSZoning'].map( {'A': 0, 'C': 1, 'FV': 2,'I':3,'RH':4,'RL':5,'RP':6,'RM':7} ).astype(float).fillna(0)

for dataset in combine:
    dataset['BldgType'] = dataset['BldgType'].map( {'1Fam': 0, '2FmCon': 1, 'Duplx': 2,'TwnhsE':3,'TwnhsI':4} ).astype(float).fillna(0)
for dataset in combine:
    dataset['SaleType'] = dataset['SaleType'].map( {'WD': 0, 'CWD': 1, 'VWD': 2,'New':3,'COD':4,'Con':5,'ConLw':6,'ConLI':7,'ConLD':8,'Oth':9} ).astype(float).fillna(0)
for dataset in combine:
    dataset['SaleCondition'] = dataset['SaleCondition'].map( {'Normal': 0, 'Abnormal': 1, 'AdjLand': 2,'Alloca':3,'Family':4,'Partial':5} ).astype(float).fillna(0)


for dataset in combine:
    dataset['Alley'] = dataset['Alley'].map( {'Grvl': 0, 'Pave': 1, 'NA': 2} ).astype(float).fillna(0)
for dataset in combine:
    dataset['LotShape'] = dataset['LotShape'].map( {'Reg': 0, 'IR1': 1, 'IR2': 2,'IR3':3} ).astype(float).fillna(0)
for dataset in combine:
    dataset['LotConfig'] = dataset['LotConfig'].map( {'Inside': 0, 'Corner': 1, 'CulDSac': 2,'FR2':3,'FR3':4} ).astype(float).fillna(0)
for dataset in combine:
    dataset['BldgType'] = dataset['BldgType'].map( {'1Fam': 0, '2FmCon': 1, 'Duplx': 2,'TwnhsE':3,'TwnhsI':4} ).astype(float).fillna(0)
print(train_df.columns)

#from sklearn.ensemble import RandomForestClassifier
#X_train=train_df.drop('SalePrice',axis=1)
#Y_train=train_df['SalePrice']
#print("hello")
#X_test=test_df.drop('Id',axis=1)
#random_forest = RandomForestClassifier(n_estimators=150)
#random_forest.fit(X_train, Y_train)
#Y_pred = random_forest.predict(X_test)
##print(Y_pred)
#submission = pd.DataFrame({
#        "Id": test_df['Id'],
#        "SalePrice": Y_pred
#    })
#submission.to_csv('submission5.csv', index=False)