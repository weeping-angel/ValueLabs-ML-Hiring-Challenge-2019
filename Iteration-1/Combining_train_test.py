import pandas as pd

train = pd.read_csv('Train_new_shuffled.csv', header=None)
test = pd.read_csv('Test_new.csv', header=None)

print('Training Data Shape', train.shape)
print(train.head())
print('Testing Data Shape', test.shape)
print(test.head())

db = pd.concat([test, train], ignore_index=True)
print('Shape of whole db : ', db.shape)
db[1:].to_csv('DB.csv', index=None)

