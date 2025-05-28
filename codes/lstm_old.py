module_name = 'LSTM'

'''
Version: v1.3.3

Description:
    Uses LSTM model to classify objects

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 04/23/2024
Date Last Updated: 11/21/2024

Doc:
    <***>

Notes:
    <***>
'''


# OTHER IMPORTS
import config        as cfg
from config import verbose
import rnn               
from visualizer           import plot_learner
import utils_model   as utls_md
if cfg.useTestSet:
    from HPT_test import searchprmtrs
else:
    from HPT_full_v1          import searchprmtrs


# CUSTOM IMPORTS
import tensorflow        as tf
import numpy             as np
import pickle            as pckl
from sklearn.model_selection import train_test_split


# USER INTERFACE
pathInput      = "D:\\Data"
pathOutput     = "D:\\ASS3"
pathsaveweight = "C:\\D2\\MindPrint\\Code\\MindPrint\\MindPrint_1\\OUTPUT\\save_weight"
units          = [100, 60, 40]
num_hidden_lyr = len(units)
metric         = 'accuracy'
split_num      = 0.2
len_seq        = 10 
learning_rate  = 0.001
batch_size     = 8
num_epochs     = 100
dense_unit     = 20
num_classes    = 2
lngth_seq      = rnn.lngth_seq
useinCV        = False
data3D_type    = True


# CONSTANTS




# FUNCTIONS
def main():
    data = rnn.readimage(pathInput)

    # Use the LSTM
    lstm_ntwrk, assessment, assessment2, conf_mtrx = lstm_output(data, data[0].shape[2], useinCV, 'LSTM', bandID=None, name=None, 
                                                                 withrepr=False, reprtype='LSTM', data3D_type=data3D_type, pathGrp='Trial01') 
#


def lstm_model(Xtrain, Ytrain, bandID, num_feats, withrepr, reprtype, data3D_type=bool,
              model_name='LSTM', pathGrp=str):
    
    # Find the parameter
    tuner, best_hps, model_structure = searchprmtrs(Xtrain, Ytrain, model_name, bandID, None, None, 
                                                    num_feats, data3D_type, withrepr, reprtype, pathGrp=pathGrp)
    
    # Load the architecture
    ntwrk = tuner.get_best_models(num_models=1)[0]

    # Parameters
    optimizer   = best_hps.values["optimizer"]
    batch_size  = best_hps.values["batch_size"]
    num_epochs  = best_hps.values["epochs"]

    return ntwrk, best_hps, optimizer, batch_size, num_epochs
#



def lstm_learner(num_feats, Xtrain, Ytrain, Xtest, Ytest, model_name, bandID, withrepr,
                reprtype, data3D_type=bool, pathGrp=str):
    
    ntwrk, best_hps, optimizer, batch_size, num_epochs = lstm_model(Xtrain, Ytrain, bandID, num_feats, withrepr, reprtype,
                                                                    data3D_type=data3D_type, model_name='LSTM', pathGrp=pathGrp)
    
    # Compile the netork
    ntwrk.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate), loss='sparse_categorical_crossentropy',
                  metrics=[metric])
    
    # Training
    history = ntwrk.fit(Xtrain, Ytrain, batch_size = batch_size, epochs = num_epochs, verbose=verbose,
                        validation_split=split_num)
    
    # Evaluate the model
    if cfg.num_sbjs_cmb > 2:
        predictions       = ntwrk.predict(Xtest)
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predictions       = ntwrk.predict(Xtest)
        predicted_classes = np.argmax(ntwrk.predict(Xtest), axis=1)

    # See predictions
    correct_indicies   = np.nonzero(predicted_classes == Ytest)[0]
    incorrect_indicies = np.nonzero(predicted_classes != Ytest)[0]

    pathprd = utls_md.create_modelfolder(pathname=pathOutput,model_name=model_name,expid=f'{cfg.expId}\\{pathGrp}', 
                                         bandID=bandID,fldrname='PREDICTIONS')
    try:
        with open(f'{pathprd}\\probaprdtns_{model_name}.pckl','wb') as pickle_file:  
            pckl.dump(predictions, pickle_file)
        with open(f'{pathprd}\\prdtns_{model_name}.pckl','wb') as pickle_file1:  
            pckl.dump(predicted_classes, pickle_file1)
    except:
        strLog = f'\n pickle file not created for saving scores'
        print(strLog)
    
    return ntwrk, best_hps, history, predictions, predicted_classes
#



def lstm_output(img_data, num_feats, useinCV, model_name, bandID=None, name=None, withrepr=bool, 
               reprtype=str, data3D_type=True, pathGrp=str):
    """
    Outputs the model assessment and model

    Input:
    ------
    n_gridpt_x : grid size of x
    n_gridpt_y : grid size of y
    num_feats  : # of features
    data    : Entire datasets (all sessions and subjects) dataframe of [num_samples, num_feats, num_chs]
    seed_num: # of seed
    df_info : dataframe containing the sessions and classlabels
    sm      : semester
    name    : name of the stimulus

    Output:
    -------
    : Return model assessment, network and confusion matrix
    """
    
    # Get the data
    Xdata, Ydata = img_data

    # Changing the shape of the data using the number of sequence
    Xdata = Xdata.reshape(len(Xdata), lngth_seq, Xdata.shape[2])


    # Split the data
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, test_size=0.4)

    # Learner
    rnn_ntwrk, best_hps, history, pred_proba, pred_cls = lstm_learner(num_feats, Xtrain, Ytrain, Xtest, Ytest, model_name, bandID, withrepr,
                                                                     reprtype, data3D_type=data3D_type, pathGrp=pathGrp)

    # Plot of Accuracy and Loss 
    history_dict = history.history
    

    acc      = history_dict['accuracy']
    val_acc  = history_dict['val_accuracy']
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']

    epoch_range = range(1, len(acc) + 1)

    # Loss and validation loss
    # plot_learner(epoch_range, loss, val_loss, bandID, 'loss', 'RNN')

    # # See plot of accuracy
    # plot_learner(epoch_range, acc, val_acc, bandID, 'accuracy', 'RNN')

    # Model Assessement
    assessment, assessment2, conf_mtrx = rnn.model_assessment(Ytest, pred_cls, pred_proba, name, 'LSTM', bandID, useinCV, 
                                                          plot=True)
    
    return rnn_ntwrk, assessment, assessment2, conf_mtrx
#





if __name__ == "__main__":
    print(f"\"{module_name}\"module begins.")

    main()