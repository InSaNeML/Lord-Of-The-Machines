#import pandas and numpy and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#design a Scaler to scale the data into normalized form
from sklearn.preprocessing import StandardScaler

#import keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

#import to sample a part of data for testing purpose
from sklearn.model_selection import train_test_split

#import the modified data
print("Reading data file")
df = pd.read_csv("./data/modified_train.csv")

#generating sparse matrix to help better
print("Creating sparse matrix")
for x in df.communication_type.unique():
    s = "comm"+str(x)
    df[s] = (df.communication_type == x).astype(int)

for x in df.email_body.unique():
    s = "body"+str(x)
    df[s] = (df.email_body == x).astype(int)

for x in df.subject.unique():
    s = "sub"+str(x)
    df[s] = (df.subject == x).astype(int)

for x in df.email_url.unique():
    s = "url"+str(x)
    df[s] = (df.email_url == x).astype(int)

#remove these columns as we don't need them anymore
print("Modifying data.")
df.drop(['communication_type','email_url','subject','email_body'], axis=1, inplace=True)

#separate out the predictor variable and target variable from the data
X = df.drop(['is_open','is_click','id','user_id','campaign_id'], axis=1)
y = df['is_click']

print(X.head(5))
print(y.head(5))

#separate out the data into training and testing part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

#we need to scale the data now
print("Scaling the data.")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

#creating and compiling a small neural network
model = Sequential()
model.add(Dense(89,kernel_regularizer=regularizers.l2(0.01), activation="relu", input_shape=(89,)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#fit the model on the data
small_model = model.fit(X_scaled, y_train, epochs=20, validation_split=0.3, batch_size = 100)

#finding the score of the model on test data
score = model.evaluate(X_test_scaled, y_test, batch_size = 100)
print(">>>>>>>>>>>>>>>>>>>>>>Score for 1 hidden layer network : ", score)

#plotting graph
acc = small_model.history['acc']
val_acc = small_model.history['val_acc']
loss = small_model.history['loss']
val_loss = small_model.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Plot for 1 hidden layer")
plt.show()

#creating and compiling a bigger neural network
model = Sequential()
model.add(Dense(120,kernel_regularizer=regularizers.l2(0.01), activation="relu", input_shape=(89,)))
model.add(Dense(89,kernel_regularizer=regularizers.l2(0.01), activation="relu", input_shape=(89,)))
model.add(Dense(30,kernel_regularizer=regularizers.l2(0.01), activation="relu", input_shape=(89,)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#fit the model on the data
large_model = model.fit(X_scaled, y_train, epochs=20, validation_split=0.3, batch_size = 100)

#finding the score of the model on test data
score = model.evaluate(X_test_scaled, y_test, batch_size = 100)
print(">>>>>>>>>>>>>>>>>>>>>>Score for 1 hidden layer network : ", score)

#plotting graph
acc = large_model.history['acc']
val_acc = large_model.history['val_acc']
loss = large_model.history['loss']
val_loss = large_model.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Plot for 3 hidden layers")
plt.show()