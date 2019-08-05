
import csv
import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.models import BayesianModel
from pgmpy.estimators import K2Score
from pgmpy.factors.discrete import  TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination



feature_val1=pd.read_csv('15features_f.csv')
'''1pen_pressure	2letter_spacing	3size	4dimension	 5is_lowercase
6is_continuous 7slantness 8tilt	9entry_stroke_a
10staff_of_a	11formation_n	 12staff_of_d	 13exit_stroke_d	 
14word_formation  15constancy'''

hill = HillClimbSearch(feature_val1, scoring_method=K2Score(feature_val1))
f_model = hill.estimate()
print(f_model.edges())

feature_val2=pd.read_csv('15features_g.csv')

hill1 = HillClimbSearch(feature_val2, scoring_method=K2Score(feature_val2))
g_model = hill1.estimate()
print(g_model.edges())


corr_mat = feature_val1.corr()
print(corr_mat)
corr_feature = set()
for i in range(len(corr_mat .columns)):  
    for j in range(i):
        if abs(corr_mat.iloc[i, j]) > 0.2:
            colname = corr_mat.columns[i]
            corr_feature.add(colname)
            
print(corr_feature) 

verify_model = BayesianModel([('pen_pressure_f', 'is_lowercase_f'), ('pen_pressure_f', 'letter_spacing_f'), ('size_f', 'slantness_f'), ('size_f', 'pen_pressure_f'), ('size_f', 'staff_of_d_f'), ('size_f', 'letter_spacing_f'), ('size_f', 'exit_stroke_d_f'), ('size_f', 'entry_stroke_a_f'), ('dimension_f', 'size_f'), ('dimension_f', 'is_continuous_f'), ('dimension_f', 'slantness_f'), ('dimension_f', 'pen_pressure_f'), ('is_lowercase_f', 'staff_of_a_f'), ('is_lowercase_f', 'exit_stroke_d_f'), ('is_continuous_f', 'exit_stroke_d_f'), ('is_continuous_f', 'letter_spacing_f'), ('is_continuous_f', 'entry_stroke_a_f'), ('is_continuous_f', 'staff_of_a_f'), ('is_continuous_f', 'is_lowercase_f'), ('slantness_f', 'is_continuous_f'), ('slantness_f', 'tilt_f'), ('entry_stroke_a_f', 'pen_pressure_f'), ('formation_n_f', 'constancy_f'), ('formation_n_f', 'word_formation_f'), ('formation_n_f', 'dimension_f'), ('formation_n_f', 'staff_of_d_f'), ('formation_n_f', 'is_continuous_f'), ('formation_n_f', 'size_f'), ('formation_n_f', 'staff_of_a_f'), ('staff_of_d_f', 'is_continuous_f'), ('staff_of_d_f', 'exit_stroke_d_f'), ('staff_of_d_f', 'is_lowercase_f'), ('staff_of_d_f', 'slantness_f'), ('staff_of_d_f', 'entry_stroke_a_f'), ('word_formation_f', 'dimension_f'), ('word_formation_f', 'staff_of_a_f'), ('word_formation_f', 'size_f'), ('word_formation_f', 'staff_of_d_f'), ('word_formation_f', 'constancy_f'), ('constancy_f', 'staff_of_a_f'), ('constancy_f', 'letter_spacing_f'), ('constancy_f', 'dimension_f'),('pen_pressure_g', 'is_lowercase_g'), ('pen_pressure_g', 'letter_spacing_g'), ('size_g', 'slantness_g'), ('size_g', 'pen_pressure_g'), ('size_g', 'staff_of_d_g'), ('size_g', 'letter_spacing_g'), ('size_g', 'exit_stroke_d_g'), ('size_g', 'entry_stroke_a_g'), ('dimension_g', 'size_g'), ('dimension_g', 'is_continuous_g'), ('dimension_g', 'slantness_g'), ('dimension_g', 'pen_pressure_g'), ('is_lowercase_g', 'staff_of_a_g'), ('is_lowercase_g', 'exit_stroke_d_g'), ('is_continuous_g', 'exit_stroke_d_g'), ('is_continuous_g', 'letter_spacing_g'), ('is_continuous_g', 'entry_stroke_a_g'), ('is_continuous_g', 'staff_of_a_g'), ('is_continuous_g', 'is_lowercase_g'), ('slantness_g', 'is_continuous_g'), ('slantness_g', 'tilt_g'), ('entry_stroke_a_g', 'pen_pressure_g'), ('formation_n_g', 'constancy_g'), ('formation_n_g', 'word_formation_g'), ('formation_n_g', 'dimension_g'), ('formation_n_g', 'staff_of_d_g'), ('formation_n_g', 'is_continuous_g'), ('formation_n_g', 'size_g'), ('formation_n_g', 'staff_of_a_g'), ('staff_of_d_g', 'is_continuous_g'), ('staff_of_d_g', 'exit_stroke_d_g'), ('staff_of_d_g', 'is_lowercase_g'), ('staff_of_d_g', 'slantness_g'), ('staff_of_d_g', 'entry_stroke_a_g'), ('word_formation_g', 'dimension_g'), ('word_formation_g', 'staff_of_a_g'), ('word_formation_g', 'size_g'), ('word_formation_g', 'staff_of_d_g'), ('word_formation_g', 'constancy_g'), ('constancy_g', 'staff_of_a_g'), ('constancy_g', 'letter_spacing_g'), ('constancy_g', 'dimension_g'),('dimension_f', 'output'),('dimension_g', 'output')]) 
print(verify_model.edges())
print(verify_model.nodes())


fields=['left', 'right', 'label']
seen_dat=pd.read_csv('dataset_seen_training_siamese_seen.csv',usecols=fields)

seen_dat['pen_pressure_f']=''
seen_dat['letter_spacing_f']=''
seen_dat['size_f']=''
seen_dat['dimension_f']=''
seen_dat['is_lowercase_f']=''
seen_dat['is_continuous_f']=''
seen_dat['slantness_f']=''
seen_dat['tilt_f']=''
seen_dat['entry_stroke_a_f']=''
seen_dat['staff_of_a_f']=''
seen_dat['formation_n_f']=''
seen_dat['staff_of_d_f']=''
seen_dat['exit_stroke_d_f']=''
seen_dat['word_formation_f']=''
seen_dat['constancy_f']=''

seen_dat['pen_pressure_g']=''
seen_dat['letter_spacing_g']=''
seen_dat['size_g']=''
seen_dat['dimension_g']=''
seen_dat['is_lowercase_g']=''
seen_dat['is_continuous_g']=''
seen_dat['slantness_g']=''
seen_dat['tilt_g']=''
seen_dat['entry_stroke_a_g']=''
seen_dat['staff_of_a_g']=''
seen_dat['formation_n_g']=''
seen_dat['staff_of_d_g']=''
seen_dat['exit_stroke_d_g']=''
seen_dat['word_formation_g']=''
seen_dat['constancy_g']=''
'''print(seen_dataset)'''
val1=[]

table_values=[]
table_values2=[]
a1=0
a2=0
a3=0
features_f=pd.read_csv('15features_f.csv')


with open('dataset_seen_training_siamese.csv', mode='r') as f:
    
    table_values2= [row for row in csv.reader(f)]
    name= seen_dat['left']
with open('15features_f.csv', mode='r') as g:
    
    table_values= [row for row in csv.reader(g)]
    name1= features_f['imagename']
    
    for i in name:
        a1=a1+1
        
        for j in name1:
            a2=a2+1
            
            if i==j:
                a=[]
                a=feature_val1[a2][1:16]
                seen_dat.iloc[a1,3:18]=a
                
                print(seen_dat)
                a2=0
                break

            if i not in j:
                a3=a3+1
                print('a3',a3)
                a2=0
                seen_dat.drop(a1)
            
     
table_values=[]
table_values1=[]
a1=0
a2=0
a3=0

with open('dataset_seen_training_siamese.csv', mode='r') as f:
    
    table_values= [row for row in csv.reader(f)]
    name= seen_dat['right']
with open('15features_g.csv', mode='r') as g:
    
    features_g=pd.read_csv('15features_g.csv') 
    table_values1= [row for row in csv.reader(g)]
    name1= features_g['imagename']
    
    for i in name:
        a1=a1+1
        
        for j in name1:
            a2=a2+1
            
            if i==j:
                a=[]
                a=table_values[a2][1:16]
                seen_dat.iloc[a1,18:33]=a
                print(seen_dat)
                a2=0
                break

            if i not in j:
                a2=0
                seen_dat.drop(a1)
print(seen_dat)
sd=seen_dat.iloc[1:,2]  
verify_model.fit(seen_dat,sd)
inference = VariableElimination(verify_model)
inference.induced_graph(['pen_pressure_f' ,	 'letter_spacing_f' , 'size_f',	'dimension_f',	 'is_lowercase_f', 'is_continuous_f' ,'slantness_f' ,'tilt_f' , 'entry_stroke_a_f', 'staff_of_a_f'	,'formation_n_f'	,' staff_of_d_f', 'exit_stroke_d_f	' ,'word_formation',  'constancy'])
phi_query = inference.map_query(variables=['pen_pressure_g'] , evidence={'pen_pressure_f':1 ,	 'letter_spacing_g':1 , 'size_f':2 ,	'is_lowercase_f': 2, 'slantness_f' : 0,'tilt_g': 1 , 'entry_stroke_a_f': 1, 'staff_of_a_f'	:0,'formation_n_f':1	})
print(phi_query)
val_dataset=pd.read_csv('dataset_seen_validation_siamese.csv')
val1= val_dataset['label']
r=0
for i in val1:
    if i==phi_query:
      r=r+1      
accuracy=(r/905)*100
print("Seen dataset accuracy:")
print(accuracy)



      

'--------------------------------------------------------------------'
f_val=pd.read_csv('15features_f.csv')
'''1pen_pressure	2letter_spacing	3size	4dimension	 5is_lowercase
6is_continuous 7slantness 8tilt	9entry_stroke_a
10staff_of_a	11formation_n	 12staff_of_d	 13exit_stroke_d	 
14word_formation  15constancy'''

hc1 = HillClimbSearch(f_val, scoring_method=K2Score(f_val))
f_model = hc1.estimate()
print(f_model.edges())

f1_val=pd.read_csv('15features_g.csv')

hc2 = HillClimbSearch(f1_val, scoring_method=K2Score(f1_val))
g_model = hc2.estimate()
print(g_model.edges())


correlation_matrix = f_val.corr()
print(correlation_matrix)
correlated_features = set()
for i in range(len(correlation_matrix .columns)):  
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.2:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
            
print(correlated_features) 

verify_model1 = BayesianModel([('pen_pressure_f', 'is_lowercase_f'), ('pen_pressure_f', 'letter_spacing_f'), ('size_f', 'slantness_f'), ('size_f', 'pen_pressure_f'), ('size_f', 'staff_of_d_f'), ('size_f', 'letter_spacing_f'), ('size_f', 'exit_stroke_d_f'), ('size_f', 'entry_stroke_a_f'), ('dimension_f', 'size_f'), ('dimension_f', 'is_continuous_f'), ('dimension_f', 'slantness_f'), ('dimension_f', 'pen_pressure_f'), ('is_lowercase_f', 'staff_of_a_f'), ('is_lowercase_f', 'exit_stroke_d_f'), ('is_continuous_f', 'exit_stroke_d_f'), ('is_continuous_f', 'letter_spacing_f'), ('is_continuous_f', 'entry_stroke_a_f'), ('is_continuous_f', 'staff_of_a_f'), ('is_continuous_f', 'is_lowercase_f'), ('slantness_f', 'is_continuous_f'), ('slantness_f', 'tilt_f'), ('entry_stroke_a_f', 'pen_pressure_f'), ('formation_n_f', 'constancy_f'), ('formation_n_f', 'word_formation_f'), ('formation_n_f', 'dimension_f'), ('formation_n_f', 'staff_of_d_f'), ('formation_n_f', 'is_continuous_f'), ('formation_n_f', 'size_f'), ('formation_n_f', 'staff_of_a_f'), ('staff_of_d_f', 'is_continuous_f'), ('staff_of_d_f', 'exit_stroke_d_f'), ('staff_of_d_f', 'is_lowercase_f'), ('staff_of_d_f', 'slantness_f'), ('staff_of_d_f', 'entry_stroke_a_f'), ('word_formation_f', 'dimension_f'), ('word_formation_f', 'staff_of_a_f'), ('word_formation_f', 'size_f'), ('word_formation_f', 'staff_of_d_f'), ('word_formation_f', 'constancy_f'), ('constancy_f', 'staff_of_a_f'), ('constancy_f', 'letter_spacing_f'), ('constancy_f', 'dimension_f'),('pen_pressure_g', 'is_lowercase_g'), ('pen_pressure_g', 'letter_spacing_g'), ('size_g', 'slantness_g'), ('size_g', 'pen_pressure_g'), ('size_g', 'staff_of_d_g'), ('size_g', 'letter_spacing_g'), ('size_g', 'exit_stroke_d_g'), ('size_g', 'entry_stroke_a_g'), ('dimension_g', 'size_g'), ('dimension_g', 'is_continuous_g'), ('dimension_g', 'slantness_g'), ('dimension_g', 'pen_pressure_g'), ('is_lowercase_g', 'staff_of_a_g'), ('is_lowercase_g', 'exit_stroke_d_g'), ('is_continuous_g', 'exit_stroke_d_g'), ('is_continuous_g', 'letter_spacing_g'), ('is_continuous_g', 'entry_stroke_a_g'), ('is_continuous_g', 'staff_of_a_g'), ('is_continuous_g', 'is_lowercase_g'), ('slantness_g', 'is_continuous_g'), ('slantness_g', 'tilt_g'), ('entry_stroke_a_g', 'pen_pressure_g'), ('formation_n_g', 'constancy_g'), ('formation_n_g', 'word_formation_g'), ('formation_n_g', 'dimension_g'), ('formation_n_g', 'staff_of_d_g'), ('formation_n_g', 'is_continuous_g'), ('formation_n_g', 'size_g'), ('formation_n_g', 'staff_of_a_g'), ('staff_of_d_g', 'is_continuous_g'), ('staff_of_d_g', 'exit_stroke_d_g'), ('staff_of_d_g', 'is_lowercase_g'), ('staff_of_d_g', 'slantness_g'), ('staff_of_d_g', 'entry_stroke_a_g'), ('word_formation_g', 'dimension_g'), ('word_formation_g', 'staff_of_a_g'), ('word_formation_g', 'size_g'), ('word_formation_g', 'staff_of_d_g'), ('word_formation_g', 'constancy_g'), ('constancy_g', 'staff_of_a_g'), ('constancy_g', 'letter_spacing_g'), ('constancy_g', 'dimension_g'),('dimension_f', 'output'),('dimension_g', 'output')]) 
print(verify_model1.edges())
print(verify_model1.nodes())


fields=['left', 'right', 'label']
unseen_dataset=pd.read_csv('dataset_unseen_training_siamese.csv',usecols=fields)

unseen_dataset['pen_pressure_f']=''
unseen_dataset['letter_spacing_f']=''
unseen_dataset['size_f']=''
unseen_dataset['dimension_f']=''
unseen_dataset['is_lowercase_f']=''
unseen_dataset['is_continuous_f']=''
unseen_dataset['slantness_f']=''
unseen_dataset['tilt_f']=''
unseen_dataset['entry_stroke_a_f']=''
unseen_dataset['staff_of_a_f']=''
unseen_dataset['formation_n_f']=''
unseen_dataset['staff_of_d_f']=''
unseen_dataset['exit_stroke_d_f']=''
unseen_dataset['word_formation_f']=''
unseen_dataset['constancy_f']=''

unseen_dataset['pen_pressure_g']=''
unseen_dataset['letter_spacing_g']=''
unseen_dataset['size_g']=''
unseen_dataset['dimension_g']=''
unseen_dataset['is_lowercase_g']=''
unseen_dataset['is_continuous_g']=''
unseen_dataset['slantness_g']=''
unseen_dataset['tilt_g']=''
unseen_dataset['entry_stroke_a_g']=''
unseen_dataset['staff_of_a_g']=''
unseen_dataset['formation_n_g']=''
unseen_dataset['staff_of_d_g']=''
unseen_dataset['exit_stroke_d_g']=''
unseen_dataset['word_formation_g']=''
unseen_dataset['constancy_g']=''
'''print(unseen_dataset)'''
val1=[]

table_values=[]
i1=0
i2=0
i3=0
features_f=pd.read_csv('15features_f.csv')


with open('dataset_unseen_training_siamese.csv', mode='r') as f:
    
    table_values2= [row for row in csv.reader(f)]
    name= unseen_dataset['left']
with open('15features_f.csv', mode='r') as g:
    
    table_values= [row for row in csv.reader(g)]
    name1= features_f['imagename']
    
for i in name:
        i1=i1+1
        
        for j in name1:
            i2=i2+1
            
            if i==j:
                a=[]
                a=table_values[i2][1:16]
                unseen_dataset.iloc[i1,3:18]=a
                print(unseen_dataset)
                i2=0
                break

            if i not in j:
                i3=i3+1
                print('a3',a3)
                i2=0
                unseen_dataset.drop(i1)
      
        
table_values=[]
i1=0
i2=0

with open('dataset_unseen_training_siamese.csv', mode='r') as f:
    
    table_values= [row for row in csv.reader(f)]
    name= unseen_dataset['right']
with open('15features_g.csv', mode='r') as g:
    
    features_g=pd.read_csv('15features_g.csv') 
    table_values= [row for row in csv.reader(g)]
    name1= features_g['imagename']
    
    for i in name:
        i1=i1+1
        
        for j in name1:
            i2=i2+1
            
            if i==j:
                a=[]
                a=table_values[i2][1:16]
                unseen_dataset.iloc[i1,18:33]=a
                print(unseen_dataset)
                i2=0
                break

            if i not in j:
                i2=0
                unseen_dataset.drop(i1)

print(unseen_dataset)
sd=unseen_dataset.iloc[1:,2]  
verify_model1.fit(unseen_dataset,sd)
inference = VariableElimination(verify_model1)
inference.induced_graph(['pen_pressure_f' ,	 'letter_spacing_f' , 'size_f',	'dimension_f',	 'is_lowercase_f', 'is_continuous_f' ,'slantness_f' ,'tilt_f' , 'entry_stroke_a_f', 'staff_of_a_f'	,'formation_n_f'	,' staff_of_d_f', 'exit_stroke_d_f	' ,'word_formation',  'constancy'])
phi_query = inference.map_query(variables=['pen_pressure_g'] , evidence={'pen_pressure_f':1 ,	 'letter_spacing_g':1 , 'size_f':2 ,	'is_lowercase_f': 2, 'slantness_f' : 0,'tilt_g': 1 , 'entry_stroke_a_f': 1, 'staff_of_a_f'	:0,'formation_n_f':1	})
print(phi_query)
val_dataset=pd.read_csv('dataset_unseen_validation_siamese.csv')
val1= val_dataset['label']
r=0
for i in val1:
    if i==phi_query:
      r=r+1      
accuracy=(r/129923)*100
print("Unseen dataset accuracy:")
print(accuracy)







print('-------------------------------------------------------------------')

f_values2=pd.read_csv('15features_f.csv')
'''1pen_pressure	2letter_spacing	3size	4dimension	 5is_lowercase
6is_continuous 7slantness 8tilt	9entry_stroke_a
10staff_of_a	11formation_n	 12staff_of_d	 13exit_stroke_d	 
14word_formation  15constancy'''

hc = HillClimbSearch(f_values2, scoring_method=K2Score(f_values2))
f_model = hc.estimate()
print(f_model.edges())

table_values2=pd.read_csv('15features_g.csv')

hc = HillClimbSearch(table_values2, scoring_method=K2Score(table_values2))
g_model = hc.estimate()
print(g_model.edges())


c_matrix1 = f_values2.corr()
print(c_matrix1)
correlated_features = set()
for i in range(len(c_matrix1 .columns)):  
    for j in range(i):
        if abs(c_matrix1.iloc[i, j]) > 0.2:
            colname = c_matrix1.columns[i]
            correlated_features.add(colname)
            
print(correlated_features) 

verify_model3 = BayesianModel([('pen_pressure_f', 'is_lowercase_f'), ('pen_pressure_f', 'letter_spacing_f'), ('size_f', 'slantness_f'), ('size_f', 'pen_pressure_f'), ('size_f', 'staff_of_d_f'), ('size_f', 'letter_spacing_f'), ('size_f', 'exit_stroke_d_f'), ('size_f', 'entry_stroke_a_f'), ('dimension_f', 'size_f'), ('dimension_f', 'is_continuous_f'), ('dimension_f', 'slantness_f'), ('dimension_f', 'pen_pressure_f'), ('is_lowercase_f', 'staff_of_a_f'), ('is_lowercase_f', 'exit_stroke_d_f'), ('is_continuous_f', 'exit_stroke_d_f'), ('is_continuous_f', 'letter_spacing_f'), ('is_continuous_f', 'entry_stroke_a_f'), ('is_continuous_f', 'staff_of_a_f'), ('is_continuous_f', 'is_lowercase_f'), ('slantness_f', 'is_continuous_f'), ('slantness_f', 'tilt_f'), ('entry_stroke_a_f', 'pen_pressure_f'), ('formation_n_f', 'constancy_f'), ('formation_n_f', 'word_formation_f'), ('formation_n_f', 'dimension_f'), ('formation_n_f', 'staff_of_d_f'), ('formation_n_f', 'is_continuous_f'), ('formation_n_f', 'size_f'), ('formation_n_f', 'staff_of_a_f'), ('staff_of_d_f', 'is_continuous_f'), ('staff_of_d_f', 'exit_stroke_d_f'), ('staff_of_d_f', 'is_lowercase_f'), ('staff_of_d_f', 'slantness_f'), ('staff_of_d_f', 'entry_stroke_a_f'), ('word_formation_f', 'dimension_f'), ('word_formation_f', 'staff_of_a_f'), ('word_formation_f', 'size_f'), ('word_formation_f', 'staff_of_d_f'), ('word_formation_f', 'constancy_f'), ('constancy_f', 'staff_of_a_f'), ('constancy_f', 'letter_spacing_f'), ('constancy_f', 'dimension_f'),('pen_pressure_g', 'is_lowercase_g'), ('pen_pressure_g', 'letter_spacing_g'), ('size_g', 'slantness_g'), ('size_g', 'pen_pressure_g'), ('size_g', 'staff_of_d_g'), ('size_g', 'letter_spacing_g'), ('size_g', 'exit_stroke_d_g'), ('size_g', 'entry_stroke_a_g'), ('dimension_g', 'size_g'), ('dimension_g', 'is_continuous_g'), ('dimension_g', 'slantness_g'), ('dimension_g', 'pen_pressure_g'), ('is_lowercase_g', 'staff_of_a_g'), ('is_lowercase_g', 'exit_stroke_d_g'), ('is_continuous_g', 'exit_stroke_d_g'), ('is_continuous_g', 'letter_spacing_g'), ('is_continuous_g', 'entry_stroke_a_g'), ('is_continuous_g', 'staff_of_a_g'), ('is_continuous_g', 'is_lowercase_g'), ('slantness_g', 'is_continuous_g'), ('slantness_g', 'tilt_g'), ('entry_stroke_a_g', 'pen_pressure_g'), ('formation_n_g', 'constancy_g'), ('formation_n_g', 'word_formation_g'), ('formation_n_g', 'dimension_g'), ('formation_n_g', 'staff_of_d_g'), ('formation_n_g', 'is_continuous_g'), ('formation_n_g', 'size_g'), ('formation_n_g', 'staff_of_a_g'), ('staff_of_d_g', 'is_continuous_g'), ('staff_of_d_g', 'exit_stroke_d_g'), ('staff_of_d_g', 'is_lowercase_g'), ('staff_of_d_g', 'slantness_g'), ('staff_of_d_g', 'entry_stroke_a_g'), ('word_formation_g', 'dimension_g'), ('word_formation_g', 'staff_of_a_g'), ('word_formation_g', 'size_g'), ('word_formation_g', 'staff_of_d_g'), ('word_formation_g', 'constancy_g'), ('constancy_g', 'staff_of_a_g'), ('constancy_g', 'letter_spacing_g'), ('constancy_g', 'dimension_g'),('dimension_f', 'output'),('dimension_g', 'output')]) 
print(verify_model3.edges())
print(verify_model3.nodes())



fields=['left', 'right', 'label']
shuffled_dataset=pd.read_csv('dataset_shuffled_training_siamese.csv',usecols=fields)

shuffled_dataset['pen_pressure_f']=''
shuffled_dataset['letter_spacing_f']=''
shuffled_dataset['size_f']=''
shuffled_dataset['dimension_f']=''
shuffled_dataset['is_lowercase_f']=''
shuffled_dataset['is_continuous_f']=''
shuffled_dataset['slantness_f']=''
shuffled_dataset['tilt_f']=''
shuffled_dataset['entry_stroke_a_f']=''
shuffled_dataset['staff_of_a_f']=''
shuffled_dataset['formation_n_f']=''
shuffled_dataset['staff_of_d_f']=''
shuffled_dataset['exit_stroke_d_f']=''
shuffled_dataset['word_formation_f']=''
shuffled_dataset['constancy_f']=''

shuffled_dataset['pen_pressure_g']=''
shuffled_dataset['letter_spacing_g']=''
shuffled_dataset['size_g']=''
shuffled_dataset['dimension_g']=''
shuffled_dataset['is_lowercase_g']=''
shuffled_dataset['is_continuous_g']=''
shuffled_dataset['slantness_g']=''
shuffled_dataset['tilt_g']=''
shuffled_dataset['entry_stroke_a_g']=''
shuffled_dataset['staff_of_a_g']=''
shuffled_dataset['formation_n_g']=''
shuffled_dataset['staff_of_d_g']=''
shuffled_dataset['exit_stroke_d_g']=''
shuffled_dataset['word_formation_g']=''
shuffled_dataset['constancy_g']=''
'''print(shuffled_dataset)'''
val1=[]

table_values=[]
k1=0
k2=0
k3=0
features_f=pd.read_csv('15features_f.csv')


with open('dataset_shuffled_training_siamese.csv', mode='r') as f:
    
    table_values2= [row for row in csv.reader(f)]
    name= shuffled_dataset['left']
with open('15features_f.csv', mode='r') as g:
    
    table_values= [row for row in csv.reader(g)]
    name1= features_f['imagename']
    
    for i in name:
        k1=k1+1
        
        for j in name1:
            k2=k2+1
            
            if i==j:
                a=[]
                a=table_values[k2][1:16]
                shuffled_dataset.iloc[k1,3:18]=a
                print(shuffled_dataset)
                k2=0
                break

            if i not in j:
        
                k2=0
                shuffled_dataset.drop(k1)

      
        
table_values=[]
a1=0
a2=0

with open('dataset_shuffled_training_siamese.csv', mode='r') as f:
    
    table_values= [row for row in csv.reader(f)]
    name= shuffled_dataset['right']
with open('15features_g.csv', mode='r') as g:
    
    features_g=pd.read_csv('15features_g.csv') 
    table_values= [row for row in csv.reader(g)]
    name1= features_g['imagename']
    for i in name:
        a1=a1+1
        
        for j in name1:
            a2=a2+1
            
            if i==j:
                a=[]
                a=table_values[a2][1:16]
                shuffled_dataset.iloc[a1,18:33]=a
                print(shuffled_dataset)
                a2=0
                break

            if i not in j:
                a2=0
                shuffled_dataset.drop(a1)
print(shuffled_dataset)          
sd=shuffled_dataset.iloc[1:,2]  
verify_model3.fit(shuffled_dataset,sd)
inference = VariableElimination(verify_model3)
inference.induced_graph(['pen_pressure_f' ,	 'letter_spacing_f' , 'size_f',	'dimension_f',	 'is_lowercase_f', 'is_continuous_f' ,'slantness_f' ,'tilt_f' , 'entry_stroke_a_f', 'staff_of_a_f'	,'formation_n_f'	,' staff_of_d_f', 'exit_stroke_d_f	' ,'word_formation',  'constancy'])
phi_query = inference.map_query(variables=['pen_pressure_g'] , evidence={'pen_pressure_f':1 ,	 'letter_spacing_g':1 , 'size_f':2 ,	'is_lowercase_f': 2, 'slantness_f' : 0,'tilt_g': 1 , 'entry_stroke_a_f': 1, 'staff_of_a_f'	:0,'formation_n_f':1	})
print(phi_query)
val_dataset=pd.read_csv('dataset_shuffled_validation_siamese.csv')
val1= val_dataset['label']
r=0
for i in val1:
    if i==phi_query:
      r=r+1      
accuracy=(r/5287)*100
print("Shuffled dataset accuracy:")
print(accuracy)
