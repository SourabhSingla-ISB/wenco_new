# importing libraries
import pandas as pd
import numpy as np
np.random.seed(1234)  
PYTHONHASHSEED = 0
import pickle
from sklearn import preprocessing
import keras
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation



# loading the training data
train_df = pd.read_excel('training_data.xlsx', sheet_name='Combined')


# MinMax normalization
train_df['Cycle_Norm'] = train_df['Cycle']
cols_normalize = train_df.columns.difference(['Equipment_Id','Cycle','Days', 'Out_2', 'Out_3', 'Out_4', 'Out_5'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), columns=cols_normalize, 
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)

#------------------------------------------- LSTM Model Building ------------------------------------------------------------#


sequence_length = 719

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]
        
        
# required columns 
sequence_cols = list(train_df.columns.difference(['Equipment_Id','Cycle', 'Out_2', 'Out_3', 'Out_4', 'Out_5']))


# generator for the sequences
seq_gen = (list(gen_sequence(train_df[train_df['Equipment_Id']==id], sequence_length, sequence_cols)) 
           for id in train_df['Equipment_Id'].unique())


# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print("Shape of sequence array: ",seq_array.shape)


# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


# generator for labels
label_gen = [gen_labels(train_df[train_df['Equipment_Id']==id], sequence_length, ['Out_2','Out_3','Out_4','Out_5']) 
             for id in train_df['Equipment_Id'].unique()]

# generate labels and convert to numpy array
label_array = np.concatenate(label_gen).astype(np.float32)
print("Shape of sequence array: ",label_array.shape)


# build the network
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(LSTM(input_shape=(sequence_length, nb_features),units=100,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit the network
model.fit(seq_array, label_array, epochs=2, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])


# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('Accurracy: {}'.format(scores[1]))


# save the model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
