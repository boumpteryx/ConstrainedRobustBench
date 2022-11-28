# python file to load, preprocess and save the medical dataset LGB from the WIDS2020 datathon
# see https://www.kaggle.com/code/krishnakartik1/wids2020-lgb-starter-adversarial-validati-f763f7

import pandas as pd
import numpy as np
import missingno as msno

train = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")
samplesubmission = pd.read_csv("/kaggle/input/widsdatathon2020/samplesubmission.csv")
test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")
dictionary = pd.read_csv("/kaggle/input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv")
solution_template = pd.read_csv("/kaggle/input/widsdatathon2020/solution_template.csv")

print('train ' , train.shape)
print('test ' , test.shape)
print('samplesubmission ' , samplesubmission.shape)
print('solution_template ' , solution_template.shape)
print('dictionary ' , dictionary.shape)

dico = pd.DataFrame(dictionary.T.head(6))
dico.columns=list(dico.loc[dico.index == 'Variable Name'].unstack())
dico = dico.loc[dico.index != 'Variable Name']
# dico.columns
train_stat = pd.DataFrame(train.describe())
train_stat2 = pd.concat([dico,train_stat],axis=0)
# train_stat2.head(20)

# we also remove the target and the ids
to_drop = ['encounter_id', 'patient_id', 'hospital_death']
# this is a list of features that look like to be categorical
categoricals_features = ['hospital_id','ethnicity','gender','hospital_admit_source','icu_admit_source','icu_stay_type','icu_type','apache_3j_bodysystem','apache_2_bodysystem','apache_3j_diagnosis_split0']
categoricals_features = [col for col in categoricals_features if col not in to_drop]

# this is the list of all input feature we would like our model to use
features = [col for col in train.columns if col not in to_drop]
print('numerber of features ' , len(features))
print('shape of train / test ', train.shape , test.shape)

more_drop = ['hospital_id','icu_id','apache4ahospitaldeathprob', 'apache4aicudeath_prob']
features = [col for col in features if col not in more_drop]
categoricals_features = [col for col in categoricals_features if col not in more_drop]

print('Transform all String features to category.\n')
for usecol in categoricals_features:
    train[usecol] = train[usecol].astype('str')
    test[usecol] = test[usecol].astype('str')

    # Fit LabelEncoder
    le = LabelEncoder().fit(
        np.unique(train[usecol].unique().tolist() +
                  test[usecol].unique().tolist()))

    # At the end 0 will be used for null values, so we start at 1
    train[usecol] = le.transform(train[usecol]) + 1
    test[usecol] = le.transform(test[usecol]) + 1

    train[usecol] = train[usecol].replace(np.nan, 0).astype('int').astype('category')
    test[usecol] = test[usecol].replace(np.nan, 0).astype('int').astype('category')

# drop all lines that have missing values
train = train[features].dropna()
test = test[features].dropna()

# save new dataset
