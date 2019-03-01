import tensorflow as tf 
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('/home/dragonbreath/Documents/pima-indians-diabetes.csv')


#Normalizing 
columns_to_normalize = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure','Triceps', 'Insulin', 'BMI', 'Pedigree']

dataset[columns_to_normalize] = dataset[columns_to_normalize].apply(lambda x: (x - x.min())/(x.max()-x.min()))

#HANDLING NUMERICAL VALUES
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')


#Handling Categorial Column

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])

#Converting numerical to categorial column 

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])

feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,assigned_group, age_buckets]


# TRAIN TEST SPLIT

x_data = dataset.drop('Class',axis=1)
labels = dataset['Class']

X_train,X_test,y_train,y_test = train_test_split(x_data,labels,test_size=.3,random_state=101)


inputFn  = tf.estimator.inputs.pandas_input_fn(x=X_train,
y=y_train,
batch_size=10,
num_epochs=1000,
shuffle=True
)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)

model.train(input_fn=inputFn,steps=1000)


eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

results = model.evaluate(eval_input_func)

print 'Model Accuracy: ',results['accuracy']