import tensorflow as tf 
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('/home/dragonbreath/Documents/pima-indians-diabetes.csv')


#Normalizing 
columns_to_normalize = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure','Triceps', 'Insulin', 'BMI', 'Pedigree']

dataset[columns_to_normalize] = dataset[columns_to_normalize].apply(lambda x: (x - x.min())/(x.max()-x.min()))

#HANDLING NUMERICAL VALUES
featureColumns = []
for column in dataset.columns:
       if column == 'Group' or column == 'Age':
           continue
       featureColumns.append(tf.feature_column.numeric_column(column))


#Handling Categorial Column

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])

#Converting numerical to categorial column 

age_bucket = tf.feature_column.bucketized_column(featureColumns[len(featureColumns)-1],boundaries=[20,30,40,50,60,70,80])

featureColumns.append(age_bucket)

# TRAIN TEST SPLIT

x_data = dataset.drop('Class',axis=1)
labels = dataset['Class']

X_train,X_test,y_test,y_test = train_test_split(x_data,labels,test_size=.3,random_state=101)