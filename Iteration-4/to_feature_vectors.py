from feature_vector import FeatureVector
import pandas as pd

fv = FeatureVector()
df = pd.read_csv('Train_new.csv')
p=[]

for i in range(0,len(df)):
    q,a,d,l = df.iloc[i]
    vec = fv.make_vector(q,a,d)
    vec = vec + [l]
    p.append(vec)
    
fw = pd.DataFrame(p, columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','label'])
fw.to_csv('Train_vec.csv', index=None)
