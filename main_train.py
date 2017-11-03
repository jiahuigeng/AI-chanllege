import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K
from keras.models import model_from_json
from util import load_train_data, load_test_data, find_groups

train_file = "ai_challenger_stock_train_20171013/stock_train_data_20171013.csv"
test_file = "ai_challenger_stock_test_20171013/stock_test_data_20171013.csv"
save_dir = os.path.join(os.getcwd(), 'saved_models')
groups = find_groups(train_file)
for id in groups:
    train_X, train_y, test_X, test_y, train_weight, test_weight = load_train_data(train_file,id)
    model_name = 'model'+str(id)+'.json'
    weight_name = 'model'+str(id)+'.h5'

    '''   
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
    #score = model.evaluate(test_X, test_y, batch_size=32,sample_weight= test_weight)
    #print( score)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    with open(model_path) as json_file:
        json_file.write(json_file)
    weight_path = os.path.join(save_dir, weight_name)
    model.save_weights(model_path)




#predictions = model.predict()
groups = find_groups(test_file):