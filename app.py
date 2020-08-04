from flask import Flask, render_template
from flask_restful import Resource,Api
import pandas as pd
import numpy as np
np.random.seed(1234)  
PYTHONHASHSEED = 0
import pickle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from flask import jsonify
import os, uuid, sys
from azure.storage.blob import BlockBlobService, PublicAccess



# load pickle model
filename = 'model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))


# load data
test_data = pd.read_excel('test_data.xlsx',sheet_name='Combined')

# sensor details data
sensor_details = pd.read_excel('test_data.xlsx',sheet_name='sensor_details')


app = Flask(__name__)
api = Api(app)

# @app.route('/', methods=['GET','POST'])
# def main():
    # return render_template('index.htm')



#@app.route('/asset_id/<int:input_ID>/', methods=['GET'])
@app.route('/', methods=['GET'])
def fn(input_ID=None):
        
    out_str = "Assest_ID does not exist"
    
    # selecting the data for a particular asset
    data = test_data
         

    # MinMax normalization
    data['Cycle_Norm'] = data['Cycle']
    cols_normalize = data.columns.difference(['Equipment_Id','Cycle','Days','Eq-Model-Code', 'Description'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(data[cols_normalize]), 
                                 columns=cols_normalize, 
                                 index=data.index)
                                 
    join_df = data[data.columns.difference(cols_normalize)].join(norm_test_df)
    data = join_df.reindex(columns = data.columns)



    # pick a window size of 719 cycles
    sequence_length = 719

    # function to reshape features into (samples, time steps, features) 
    def gen_sequence(id_df, seq_length, seq_cols):
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]
            
            
    # required columns 
    sequence_cols = list(data.columns.difference(['Equipment_Id','Cycle','Eq-Model-Code', 'Description']))


    # generator for the sequences
    seq_gen = (list(gen_sequence(data[data['Equipment_Id']==id], sequence_length, sequence_cols)) 
               for id in data['Equipment_Id'].unique())


    # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    seq_array.shape


    # predicting the value
    pred_value = loaded_model.predict(seq_array)

    pred_value[pred_value>=0.5] = 1
    pred_value[pred_value<0.5] = 0

    
    
    # converting the output into dictionary
    # converting the output into dictionary
    d = {}
    for i, row in enumerate(pred_value):
        d[data['Equipment_Id'].unique()[i]] = [int(x) for x in row.tolist()]
                
    
    # # selecting those assets which require maintenance
    # newDict = dict()
    # if len(d) > 1:
        # for (key, value) in d.items():
            # if any(val==1 for val in value):
                # newDict[key] = value
                # d = newDict
    
    # converting dictionary into dataframe
    df = pd.DataFrame.from_dict(d, orient='index')

    # selecting only those assets which require maintainance
    df = df.loc[(df!=0).any(axis=1)]

    # Including index as column
    df.reset_index(level=0, inplace=True)

    # renaming columns
    df.columns = ['Equipment_Id', 'LeftCoolant', 'RightCoolant', 'LeftHose','RightHose']

    # replacing 0/1 with Required/Not-Required
    df.replace({0: 'Not-Required', 1: 'Required'},inplace=True)

    # merging dataframes to get sensors details
    result = pd.merge(df,sensor_details, how='left',on='Equipment_Id')

    # re-arranging the columns
    result = result[['Equipment_Id','Eq-Model-Code', 'Description','LeftCoolant', 'RightCoolant', 'LeftHose','RightHose']]

    # saving dataframe into azure blob as a csv file
    output = result.to_csv (index_label="SNo.", encoding = "utf-8")

    accountName = "wenco1"
    accountKey = "FwniZZzezkiacqf269reGr0kFdFg8vG+gIZG4uxSh7eIczYq0hHYb0+GRFBDvG/GmsK7WSLpB4hzh+dGd6AS7g=="
    containerName = "wenco1"


    blobService = BlockBlobService(account_name=accountName, account_key=accountKey)

    blobService.create_blob_from_text(containerName, 'Prediction.csv', output)
    
        
    #return jsonify(output)
    return ("Prediction file has been successfully uploaded on Azure Blob")


 
if __name__ == "__main__":
    app.run(debug=None,threaded=False)