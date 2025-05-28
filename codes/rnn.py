module_name = 'Simple RNN'

'''
Version: v1.3.4

Description:
    Uses RNN model to classify objects

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 11/23/2024
Date Last Updated: 11/20/2024

Doc:
    The input shape take 3 dimensions 
    [num_samples, num_time_stamps, num_feats]


Notes:
    <***>
'''


# CUSTOM IMPORTS
import config        as cfg
from config          import verbose
import utils_model   as utls_md
from visualizer      import plot_learner
if cfg.useTestSet:
    from HPT_test import searchprmtrs
else:
    from HPT_full_v1          import searchprmtrs


# OTHER IMPORTS
import tensorflow        as tf
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import pickle            as pckl
import os
import cv2
from sklearn             import metrics
from sklearn.model_selection import train_test_split
from imblearn.metrics        import specificity_score, geometric_mean_score
import tifffile


# USER INTERFACE
pathInput      = "D:\\Data"
pathOutput     = "D:\\ASS3"
pathsaveweight = "C:\\D2\\MindPrint\\Code\\MindPrint\\MindPrint_1\\OUTPUT\\save_weight"
expId          = cfg.expId
units          = [500, 200, 100]
lst_units      = 50
num_hidden_lyr = len(units)
metric         = 'accuracy'
split_num      = 0.2
lngth_seq      = 27
learning_rate  = 0.001 
batch_size     = 8
num_epochs     = 100
dense_unit     = 24
data3D_type    = True
dim_size       = 512
flatten_row    =  512 * 512
alldata        = {}
grpdata        = {}
alllabel       = {}
# allframes      = {}
grpdata        = {}
min_value      = {'min':None}
num_classes    = 2
useinCV        = False
groups         = ['control', 'stimulanted']
allmodelrslt   = {}
measureLst     = ['Roc_curve','F1','Accuracy','Recall','Specificity','Precision','ConfusionMtrx',
                  'Mattscr','Kappa','MicroF1','BalancedAcc','MicroAUC','GeometricM','DiagnOdd','AdjustedFmsr']


# CONSTANTS



def ratioCM(confmtrx):
    fp = confmtrx.sum(axis=0)-np.diag(confmtrx)
    fn = confmtrx.sum(axis=1)-np.diag(confmtrx)
    tp = np.diag(confmtrx)
    tn = confmtrx.sum() - (fp+fn+tp)

    # change the object type
    fp = fp.astype(float)
    fn = fn.astype(float)
    tn = tn.astype(float)

    return fp,fn,tp,tn
# 

def adjustedFmeasure(true,pred,fp,tn):
    """
    Uses the adjuested F measure for 
    multi-classification.

    Input:
    ------
    true: true labels
    pred: predicted labels

    Output:
    ------
    : Returns AGF (an array)
    """
    # confmtrx = metrics.confusion_matrix(list(true.values),pred)
    # fp,fn,tp,tn = ratioCM(confmtrx)

    # fbeta
    fbeta = metrics.fbeta_score(true,pred,average=None,beta=2)
    # fbeta = metrics.fbeta_score(list(true.values),pred,average=None,beta=2) used for multiclass

    # NPV
    npv = tn/(fp+tn)

    # InvF0.5
    InvF = (1.25*(npv*(tn/(fp+tn))))/(0.25*npv + (tn/(fp+tn)))

    # Square root
    agf = np.sqrt(fbeta*InvF)
    return agf
#

def me_md_mx(data,nameScore):
    mmm = {}
    for i in range(len(data)):
        mmm[f'{nameScore[i]}'] = np.array([np.round(np.min(data[i]),2), np.round(np.median(data[i]),2), np.round(np.max(data[i]),2)])
    
    return  mmm


def model_assessment(true, pred, pred_proba, name, model_name, bandID, useinCV, plot=bool):
    """
    Model Assessment 

    Input:
    -----
    true      : true class (1D vector or matrix)
    pred      : predicted class (1D vector)
    pred_proba: predicted class probabilities (2D vector)
    stiname   : name of stimulus
    model_name:name of model
    bandID    : name of band
    useinCV   : whether to use in CV
    plot      : whether to plot or not 
    pathGrp   : name of groups

    Output:
    -------
    Return model assessment scores
    """
    # foldID_fusion = f'\\{model_name[-1]}' if cfg.useFusion_ui else ''
    

    nameScore  = ['auc','f1scr','accscr','recalscr','specf','prescn','drscr','adjFmscr',
                  'mattscr','kappscr','mf1scr','blaccscr','micAUC_m','gmeanscr']
    subj_names = ['control', 'stimulanted']
    # subj_names = cfg.combntn[int(pathGrp[1:])]

    # if model_name[-1].isdigit():
    #     add_name = model_name[:-1]
    # else:
    #     add_name = model_name
    
    if len(groups) > 2:
        #Multi-class
        conf_mtrx   = metrics.confusion_matrix(list(true.values), pred)
        fp,fn,tp,tn = ratioCM(conf_mtrx)

        #Individual scores
        acc          = np.array([metrics.accuracy_score(list(true.values), pred)]*num_classes)
        f1           = metrics.f1_score(list(true.values), pred, average=None)
        prcsn        = metrics.precision_score(list(true.values), pred, average=None)
        recal        = metrics.recall_score(list(true.values), pred, average=None)
        specifty     = specificity_score(list(true.values), pred, average=None)
        roc_score    = metrics.roc_auc_score(list(true.values), pred_proba, average=None, multi_class='ovr')
        dor_scr      = (tp*tn)/(fp*fn)  #diagnostic_oddratio
        adjtedFmsr   = adjustedFmeasure(true,pred,fp,tn)   

        #Overall Scores
        matt_cor     = np.array([metrics.matthews_corrcoef(list(true.values),pred)]*num_classes)
        coh_kppascr  = np.array([metrics.cohen_kappa_score(list(true.values),pred)]*num_classes)
        micro_f1scr  = np.array([metrics.f1_score(list(true.values),pred, average='micro')]*num_classes)
        balnc_accscr = np.array([metrics.balanced_accuracy_score(list(true.values),pred)]*num_classes)
        micAUC       = np.array([metrics.roc_auc_score(list(true.values),pred_proba, average='micro', 
                                                       multi_class='ovr')]*num_classes)
        micGeoMean   = np.array([geometric_mean_score(list(true.values),pred, average='micro')]*num_classes)
        
        scoreLst = (roc_score,f1,acc,recal,specifty,prcsn,dor_scr,adjtedFmsr,
                    matt_cor,coh_kppascr,micro_f1scr,balnc_accscr,micAUC,micGeoMean)

        # Applying the min, median and max function
        scoreLst_m = me_md_mx(scoreLst,nameScore)
        

        # Save the scores for the 1 Fold
        mdl_scrs = (roc_score,f1,acc,recal,specifty,prcsn,conf_mtrx,matt_cor,coh_kppascr,micro_f1scr,balnc_accscr,
                    micAUC, micGeoMean, dor_scr, adjtedFmsr)

        if plot:
            if useinCV == False:
                pathsngle = utls_md.create_modelfolder(pathname=pathOutput, model_name=model_name, expid=expId, 
                                                       fldrname='MODEL_ASSMNT\\FOLD_SNGL', bandID=bandID)
            else:
                pathsngle = utls_md.create_modelfolder(pathname=pathOutput, model_name=model_name, expid=expId, 
                                                       bandID=bandID,fldrname='MODEL_ASSMNT\\FOLDS')
            # Save the scores for single fold
            for i_scr, i_mrslst in zip(range(len(mdl_scrs)),measureLst):
                try:
                    # {pathpckl}\\{add_name} replaced by pathtrialdata
                    with open(f'{pathsngle}\\{model_name}_{i_mrslst}scrs1Fold.pckl','wb') as pickle_file:  
                        pckl.dump(mdl_scrs[i_scr], pickle_file)
                except:
                      strLog = f'\n pickle file not created for saving scores'
                      print(strLog)
            assessment = pd.DataFrame({'AUC': scoreLst_m['auc'],
                                       'F1': scoreLst_m['f1scr'],
                                       'Accuracy': scoreLst_m['accscr'],
                                       'Recall': scoreLst_m['recalscr'],
                                       'Specificity':scoreLst_m['specf'],
                                       'Precision': scoreLst_m['prescn']},index=['min','median','max'])
            
            assessment2 = pd.DataFrame({'Mattcorr': scoreLst_m['mattscr'],
                                        'Kappa': scoreLst_m['kappscr'],
                                        'MicroF1': scoreLst_m['mf1scr'],
                                        'BalancedAcc': scoreLst_m['blaccscr'],
                                        'MicroAUC': scoreLst_m['micAUC_m'],
                                        'GeometricMean': scoreLst_m['gmeanscr'],
                                        'DOR': scoreLst_m['drscr'],
                                        'AdjustedFmsr': scoreLst_m['adjFmscr']},index=['min','median','max'])
            
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mtrx, 
                                                        display_labels=subj_names).plot()
            plt.title(f'Confusion Matrix:{name}_statsFeatures')
            plt.savefig(pathsngle + '\\' + f"{name}_{model_name}Cmtrx.png") 
            plt.close()
        
            return assessment, assessment2, cm_display
        
        else:
            allmodelrslt['acc'] = acc; allmodelrslt['f1'] = f1; allmodelrslt['prscn'] = prcsn; allmodelrslt['recal'] = recal
            allmodelrslt['specfty'] = specifty; allmodelrslt['auc'] = roc_score; allmodelrslt['cmtrx'] = conf_mtrx
            allmodelrslt['matt'] = matt_cor; allmodelrslt['cohkpp'] = coh_kppascr; allmodelrslt['micf1'] = micro_f1scr
            allmodelrslt['balacc'] = balnc_accscr; allmodelrslt['aucavg'] = micAUC; allmodelrslt['gmean'] = micGeoMean
            allmodelrslt['dor'] = dor_scr; allmodelrslt['adjF'] = adjtedFmsr

            return allmodelrslt

    else:
        #Binary classification
        acc         = metrics.accuracy_score(true, pred)
        f1          = metrics.f1_score(true, pred)
        prcsn       = metrics.precision_score(true, pred)
        recal       = metrics.recall_score(true, pred)
        specifty    = specificity_score(true, pred)
        roc_score   = metrics.roc_auc_score(true, pred)
        conf_mtrx   = metrics.confusion_matrix(true, pred)
        fp,fn,tp,tn = ratioCM(conf_mtrx)
        dor_scr     = (tp*tn)/(fp*fn)
        adjtedFmsr  = adjustedFmeasure(true,pred,fp,tn)

        # Overall Scores
        matt_cor     = metrics.matthews_corrcoef(true, pred)
        coh_kppascr  = metrics.cohen_kappa_score(true, pred)
        micro_f1scr  = metrics.f1_score(true, pred)
        balnc_accscr = metrics.balanced_accuracy_score(true, pred)
        micAUC       = metrics.roc_auc_score(true, pred, average='micro')
        micGeoMean   = geometric_mean_score(true, pred, average='binary')

        auc      = pd.Series([roc_score]); f1scr = pd.Series([f1]); accscr = pd.Series([acc])
        recalscr = pd.Series([recal]); specf = pd.Series([specifty]); prescn = pd.Series([prcsn])
        mattcr   = pd.Series([matt_cor]); coh_scr = pd.Series([coh_kppascr]); micf1scr = pd.Series([micro_f1scr])
        balaccscr = pd.Series([balnc_accscr]); micgeo_scr = pd.Series([micGeoMean])
        dorscr   = pd.Series([dor_scr]); adjFscr = pd.Series([adjtedFmsr]); micaucscr = pd.Series([micAUC])

        # Save the scores for the 1 Fold
        mdl_scrs = (roc_score,f1,acc,recal,specifty,prcsn,conf_mtrx)

        if plot:
            if useinCV == False:
                pathsngle = utls_md.create_modelfolder(pathname=f'{pathOutput}\\ASS3', model_name=model_name, expid=expId, 
                                                       fldrname='MODEL_ASSMNT\\FOLD_SNGL', bandID=bandID)
            else:
                pathsngle = utls_md.create_modelfolder(pathname=f'{pathOutput}\\ASS3', model_name=model_name, expid=expId, 
                                                       bandID=bandID,fldrname='MODEL_ASSMNT\\FOLDS')
            # Save the scores for single fold
            for i_scr, i_mrslst in zip(range(len(mdl_scrs)),measureLst):
                try:
                    with open(f'{pathsngle}\\{model_name}_{i_mrslst}scrs1Fold.pckl','wb') as pickle_file:  
                        pckl.dump(mdl_scrs[i_scr], pickle_file)
                except:
                      strLog = f'\n pickle file not created for saving scores'
                      print(strLog)
            assessment = pd.DataFrame({'AUC': auc,
                                       'F1': f1scr,
                                       'Accuracy': accscr,
                                       'Recall': recalscr,
                                       'Specificity':specf,
                                       'Precision': prescn})
            
            assessment2 = pd.DataFrame({'Mattcorr': mattcr,
                                        'Kappa': coh_scr,
                                        'MicroF1': micf1scr,
                                        'BalancedAcc': balaccscr,
                                        'MicroAUC':micaucscr,
                                        'GeometricMean': micgeo_scr,
                                        'DOR': dorscr,
                                        'AdjustedFmsr': adjFscr})
            
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mtrx, 
                                                        display_labels=subj_names).plot()
            plt.title(f'Confusion Matrix:{name}_statsFeatures')
            plt.savefig(pathsngle + '\\' + "Cmtrx.png")
            plt.close()
        
            return assessment, assessment2, cm_display
        
        else:
            return acc, f1, prcsn, recal, specifty, roc_score, conf_mtrx
#



# FUNCTIONS
def main():
    
    # Read the image data
    data = readimage(pathInput)

    # Use the RNN
    rnn_ntwrk, assessment, assessment2, conf_mtrx = rnn_output(data, data[0].shape[2], useinCV, 'RNN', bandID=None, name=None, 
                                                               withrepr=False, reprtype='RNN', data3D_type=data3D_type, pathGrp='Trial01')

#


def chop_frames(dict_v):
    """
    This chops of the frames and takes the minimum number of
    frames 
    """
    save_numframes = []
    for key, matrix in dict_v.items():
        num_frames = len(dict_v[key])

        save_numframes.append(num_frames)
    # for

    min_numframes = np.min(save_numframes)

    # Chop the number of frames
    data = {key: {key1: matrix[key1] for key1 in list(matrix.keys())[:min_numframes]} 
            for key, matrix in dict_v.items()}
    
    # allnew = np.full((min_numframes, flatten_row), np.nan, dtype=np.int32)

    for idx, key in enumerate(data.keys()):
        new_data = np.full((len(data), min_numframes, flatten_row), np.nan, dtype=np.int32)
        for key1 in list(data[key].keys()):
            new_data[idx, int(key1),:] = data[key][key1]

    # Put the values into one dictionary
    # new_data = {key: map(lambda x: allnew[int(key1),:], data[key][key1]) for key in data.keys() for key1 in list(data[key].keys())}


    # data = {key: dict_v[key][:,:use_min] for key in dict_v.keys()}

    return new_data



def readimage(path):

    for grp_idx, cls_idx in zip(os.listdir(path), range(0,2)):
        # grpdata = {}
        grp_path = os.path.join(path, grp_idx)

        # Get the number of folders in the group
        num_fldrs = len(os.listdir(grp_path))

        # sort the folder names
        strd_fldr = [name for name in sorted(os.listdir(grp_path), key=lambda x:int(x.replace('S','')))]

        # Create an empty array with nans
        label_data = np.full((len(strd_fldr),1), np.nan, dtype=np.int8)

        for fldr_nme, fldr_idx in zip(strd_fldr, range(1, num_fldrs+1)):
            allframes = {}
            path_u = os.path.join(grp_path, fldr_nme)

            with tifffile.TiffFile(f'{path_u}\\ZS-{fldr_idx}.tif') as tif:
                for pge_idx, page in enumerate(tif.pages):
                    read_imgframe = page.asarray()

                    # Save the different frames
                    allframes[f'{pge_idx}'] = read_imgframe.flatten()

            # # concatenate the files into one
            # image_data = np.concatenate(list(allframes.values()), axis=1)

            # Save the data
            grpdata[f'{grp_idx}_{fldr_idx}'] = allframes
            label_data[fldr_idx-1,:]         = cls_idx
        # for

        alllabel[f'{grp_idx}'] = label_data

    # Save all the data
    # Chop some of the data off
    data = chop_frames(grpdata)

    # for

    # Concatenate the data
    label    = np.concatenate(list(alllabel.values()), axis=0)
    return [data, label]
#

def rnn_model(Xtrain, Ytrain, bandID, num_feats, withrepr, reprtype, data3D_type=bool,
              model_name='RNN', pathGrp=str):
    
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


# def myrnn(num_hidden_lyr, num_feats, len_seq=lngth_seq, data3D_type=bool):
#     """
#     A simple RNN model

#     Input:
#     -----
#     n_gridpt_x    : grid size of x
#     n_gridpt_y    : grid size of y
#     num_hidden_lyr: # of hidden layer
#     num_feats     : # of features used 
#     len_seq       : length of the time samples

#     Output:
#     -------
#     : Returns rnn model
#     """
    
#     # Input 
#     if data3D_type:
#         # the height is treated like the time
#         # input_img = tf.keras.Input(shape=(n_gridpt_x, n_gridpt_y))
#         input_img = tf.keras.Input(shape=(len_seq, num_feats))

#     else:
#         if num_feats != None:
#             input_img = tf.keras.Input(shape = (len_seq, num_feats))

#     #
#     model_ = input_img

#     for i_lyr, i_units in zip(range(num_hidden_lyr), units):
#         model_ = tf.keras.layers.SimpleRNN(i_units, return_sequences=True)(model_)
        
#     model_ = tf.keras.layers.SimpleRNN(lst_units, return_sequences=False,
#                                        activation='relu')(model_)

#     # # Flatten
#     # model_ = tf.keras.layers.Flatten()(model_)

#     # Dense Layer
#     model_ = tf.keras.layers.Dense(dense_unit, activation='relu')(model_)

#     # Last lyr
#     last_lyr = tf.keras.layers.Dense(num_classes, activation='softmax')(model_)

#     simple_rnn_ntwrk = tf.keras.Model(inputs=input_img, outputs=last_lyr)

#     # Model summary
#     simple_rnn_ntwrk.summary()

#     return simple_rnn_ntwrk
# #

def rnn_learner(num_feats, Xtrain, Ytrain, Xtest, Ytest, model_name, bandID, withrepr,
                reprtype, data3D_type=bool, pathGrp=str):
    
    ntwrk, best_hps, optimizer, batch_size, num_epochs = rnn_model(Xtrain, Ytrain, bandID, num_feats, withrepr, reprtype,
                                                                   data3D_type=data3D_type, model_name='RNN', pathGrp=pathGrp)
    
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
    

# def rnn_learner(num_feats, Xtrain, Ytrain, Xtest, Ytest, batch_size,
#                 sm, savemodel=bool, data3D_type=bool):
#     """
#     Uses the RNN model to classify subjects

#     Input:
#     ------
#     n_gridpt_x : grid size of x
#     n_gridpt_y : grid size of y
#     num_feats  : # of features
#     Xtrain, Ytrain, Xtest, Ytest : The datasets from the split
#     sm       : semester
#     batch_size : number of batch_size
#     savemodel: Whether to save the model 

#     Outputs:
#     --------
#     : Returns model, model_history and predicted class
#     """

#     # THE MODEL
#     simple_rnn_ntwrk = myrnn(num_hidden_lyr, num_feats, data3D_type=data3D_type)

#     # Compile the netork
#     simple_rnn_ntwrk.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate), loss='sparse_categorical_crossentropy',
#                              metrics=[metric])
    
#     # Training
#     history = simple_rnn_ntwrk.fit(Xtrain, Ytrain, batch_size = batch_size, epochs = num_epochs, verbose=verbose,
#                                    validation_split=split_num)
    
#     if savemodel:

#         # Save the model weights
#         simple_rnn_ntwrk.save(pathsaveweight + f"{sm}_RNN.keras")

#     # Evaluate the model
#     predtns  = simple_rnn_ntwrk.predict(Xtest)
#     pred_cls = np.argmax(predtns, axis=1)

#     pred_proba = np.max(predtns, axis=1)

#     # See predictions
#     correct_indicies   = np.nonzero(pred_cls == Ytest)[0]
#     incorrect_indicies = np.nonzero(pred_cls != Ytest)[0] 

#     return simple_rnn_ntwrk, history, pred_proba, pred_cls
# #



def rnn_output(img_data, num_feats, useinCV, model_name, bandID=None, name=None, withrepr=bool, 
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
    rnn_ntwrk, best_hps, history, pred_proba, pred_cls = rnn_learner(num_feats, Xtrain, Ytrain, Xtest, Ytest, model_name, bandID, withrepr,
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
    assessment, assessment2, conf_mtrx = model_assessment(Ytest, pred_cls, pred_proba, name, 'RNN', bandID, useinCV, 
                                                          plot=True)
    
    return rnn_ntwrk, assessment, assessment2, conf_mtrx
#



if __name__ == "__main__":
    print(f"\"{module_name}\"module begins.")

    main()