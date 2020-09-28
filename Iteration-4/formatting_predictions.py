import pandas as pd

results = pd.read_csv('Results.csv')
predictions = pd.read_csv('Predictions.csv', header=None)
preds = []
print('Total Predictions = ', len(predictions))
for i in range(0, len(predictions)):
    l, d1, d2, d3 = predictions.iloc[i]
    preds.append(','.join([str(d1),str(d2),str(d3)]))

results['distractor']=preds
results.to_csv('Predictions_new.csv', index=None)


