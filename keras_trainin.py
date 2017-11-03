import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K
from util import load_train_data, load_test_data


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'trained_model1.h5'


train_X, train_y, test_X, test_y, train_weight, test_weight = load_train_data("ai_challenger_stock_train_20171013/stock_train_data_20171013.csv",2)
#test_X = load_test_data("ai_challenger_stock_test_20171013/stock_test_data_20171013.csv",2)

'''
model = Sequential()
model.add(Dense(32,activation='relu', input_dim=88))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0,5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0,5))
model.add(Dense(1,activation='sigmoid'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9,nesterov=True)
model.compile(loss = 'binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_X,train_y,epochs=20, batch_size=128)
'''

model = Sequential()
model.add(Dense(50, input_dim=88, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_X, train_y,
          epochs=200,
          batch_size=32,shuffle=True, sample_weight= train_weight)
score = model.evaluate(test_X, test_y, batch_size=32,sample_weight= test_weight)
print( score)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('saved trained model at %s' % model_path)

#predictions = model.predict()