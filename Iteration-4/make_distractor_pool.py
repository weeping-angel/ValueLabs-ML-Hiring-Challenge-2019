import pandas as pd
import re

dists = pd.read_csv('Train.csv')['distractor']
fw = open('distractor_pool', 'w', encoding='utf-8')

for dist in dists:
    l = re.split(r'[\'\"],', dist)
    for i in l:
        i=i.strip(" ")
        i=i.strip("'")
        i=i.strip('"')
        fw.write(i.strip(' ')+"\n")
