import csv
from sets import Set
import numpy as np
from itertools import groupby
import numpy as np
def data_split(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            with open("%s.csv" % row['group'], 'a') as output:
                for i in range(88):
                    output.write(row["feature%d" %i])
                    output.write(",")
                output.write(row["weight"])
                output.write(",")
                output.write(row["era"])
                output.write(",")
                output.write(row["label"])
                output.write("\n")

def load_train_data(filename, group_id):
    rate = 0.8
    data_X =[]
    data_y = []
    weights = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['group'])==group_id:
                gfeature = []
                glabel = []
                for i in range(88):
                    if row['feature%s' %i]== '':
                        gfeature.append(0.0)
                    else:
                        gfeature.append(float(row['feature%d' %i]))
                glabel.append(float(row['label']))
                #gfeature.append(float(row['era']))
                gfeature = np.nan_to_num(gfeature)
                data_X.append(gfeature)
                data_y.append(glabel)
                weights.append(float(row['weight']))
    data_X = np.array(data_X)
    data_y = np.array(data_y)
    weights = np.array(weights)
    len_X = data_X.shape[0]
    train_X, test_X = data_X[:int(len_X*rate)],data_X[int(len_X*rate):]
    train_y, test_y = data_y[:int(len_X*rate)],data_y[int(len_X*rate):]
    train_weight, test_weight = weights[int(len_X*rate):],weights[:int(len_X*rate)]
    return train_X, train_y, test_X, test_y, train_weight, test_weight



def load_test_data(filename, group_id):
    test_X =[]
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['group'])==group_id:
                gfeature = []
                for i in range(88):
                    if row['feature%s' %i]== '':
                        gfeature.append(0.0)
                    else:
                        gfeature.append(float(row['feature%d' %i]))
                gfeature = np.nan_to_num(gfeature)
                test_X.append(gfeature)
    test_X = np.array(test_X)
    return test_X
        
def find_groups(filename):
    groups = Set()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            groups.add(int(row['group']))
    return groups




if __name__ == '__main__':
    #data_loader("ai_challenger_stock_train_20171013/stock_train_data_20171013.csv") #
    #train_X, train_y, test_X, test_y = load_train_data("ai_challenger_stock_train_20171013/stock_train_data_20171013.csv",1)
   # test_X = load_test_data("ai_challenger_stock_test_20171013/stock_test_data_20171013.csv",1)
    #print(train_X.shape)
    #print(train_y.shape)
    #print(test_X.shape)
    #print(test_y.shape)
    print(find_groups("ai_challenger_stock_train_20171013/stock_train_data_20171013.csv"))
    #print(test_X.shape)

