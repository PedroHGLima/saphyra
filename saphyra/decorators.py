
__all__ = ["Summary", "Reference", "Relevance"]


from Gaugi import Logger, StatusCode
from Gaugi.messenger.macros import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve

import numpy as np
from copy import copy
import collections

def sp_func(pd, fa):
  return np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )


#
# Decorate the history dictionary after the training phase with some useful controll values
#
class Summary( Logger ):

  #
  # Constructor
  #
  def __init__( self ):
    Logger.__init__(self)

  #
  # Use this method to decorate the keras history in the end of the training
  #
  def decorate( self, history, context ):

    d = {}
    x_train, y_train = context.getHandler("trnData")
    x_val, y_val     = context.getHandler("valData")
    model            = context.getHandler( "model" )


    # Get the number of events for each set (train/val). Can be used to approx the number of
    # passed events in pd/fa analysis. Use this to integrate values (approx)
    sgn_total = len( y_train[y_train==1] )
    bkg_total = len( y_train[y_train==0] )
    sgn_total_val = len( y_val[y_val==1] )
    bkg_total_val = len( y_val[y_val==0] )


    MSG_INFO( self, "Starting the train summary..." )

    y_pred = model.predict( x_train )
    y_pred_val = model.predict( x_val )

    # get vectors for operation mode (train+val)
    y_pred_operation = np.concatenate( (y_pred, y_pred_val), axis=0)
    y_operation = np.concatenate((y_train,y_val), axis=0)

    d['rocs'] = {}
    d['hists'] = {}

    # No threshold is needed
    d['auc'] = roc_auc_score(y_train, y_pred)
    d['auc_val'] = roc_auc_score(y_val, y_pred_val)
    d['auc_op'] = roc_auc_score(y_operation, y_pred_operation)


    # No threshold is needed
    d['mse'] = mean_squared_error(y_train, y_pred)
    d['mse_val'] = mean_squared_error(y_val, y_pred_val)
    d['mse_op'] = mean_squared_error(y_operation, y_pred_operation)



    # Here, the threshold is variable and the best values will
    # be setted by the max sp value found in hte roc curve
    # Training
    fa, pd, thresholds = roc_curve(y_train, y_pred)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = np.argmax(sp)
    threshold = thresholds[knee]


    d['rocs']['roc'] = (pd, fa)
    #d['rocs']['predictions'] = (y_pred, y_train)

    d['hists']['trn_sgn'] = np.histogram(y_pred[y_train == 1], bins=70)
    d['hists']['trn_bkg'] = np.histogram(y_pred[y_train != 1], bins=70)

    MSG_INFO( self, "Train samples     : Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
        pd[knee], fa[knee], sp[knee], d['auc'], d['mse'])


    d['max_sp_pd'] = ( pd[knee], int(pd[knee]*sgn_total), sgn_total)
    d['max_sp_fa'] = ( fa[knee], int(fa[knee]*bkg_total), bkg_total)
    d['max_sp']    = sp[knee]
    d['acc']       = accuracy_score(y_train,y_pred>threshold)

    # Validation
    fa, pd, thresholds = roc_curve(y_val, y_pred_val)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = np.argmax(sp)
    threshold = thresholds[knee]

    d['rocs']['roc_val'] = (pd, fa)
    #d['rocs']['predictions_val'] = (y_pred_val, y_val)

    d['hists']['val_sgn'] = np.histogram(y_pred_val[y_val == 1], bins=70)
    d['hists']['val_bkg'] = np.histogram(y_pred_val[y_val != 1], bins=70)

    MSG_INFO( self, "Validation Samples: Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
        pd[knee], fa[knee], sp[knee], d['auc_val'], d['mse_val'])


    d['max_sp_pd_val'] = (pd[knee], int(pd[knee]*sgn_total_val), sgn_total_val)
    d['max_sp_fa_val'] = (fa[knee], int(fa[knee]*bkg_total_val), bkg_total_val)
    d['max_sp_val']    = sp[knee]
    d['acc_val']       = accuracy_score(y_val,y_pred_val>threshold)

    # Operation
    fa, pd, thresholds = roc_curve(y_operation, y_pred_operation)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = np.argmax(sp)
    threshold = thresholds[knee]

    d['rocs']['roc_op'] = (pd, fa)
    # We dont need to attach y_op and y_pred_op since the user can concatenate train and val to get this. Just to save storage.
    #d['rocs']['predictions_op'] = (y_pred_operation, y_operations)
    
    d['hists']['op_sgn'] = np.histogram(y_pred_operation[y_operation == 1], bins=70)
    d['hists']['op_bkg'] = np.histogram(y_pred_operation[y_operation != 1], bins=70)

    MSG_INFO( self, "Operation Samples : Prob. det (%1.4f), False Alarm (%1.4f), SP (%1.4f), AUC (%1.4f) and MSE (%1.4f)",
        pd[knee], fa[knee], sp[knee], d['auc_val'], d['mse_val'])

    d['threshold_op'] = threshold
    d['max_sp_pd_op'] = ( pd[knee], int( pd[knee]*(sgn_total+sgn_total_val)), (sgn_total+sgn_total_val))
    d['max_sp_fa_op'] = ( fa[knee], int( fa[knee]*(bkg_total+bkg_total_val)), (bkg_total+bkg_total_val))
    d['max_sp_op'] = sp[knee]
    d['acc_op']              = accuracy_score(y_operation,y_pred_operation>threshold)

    history['summary'] = d

    return StatusCode.SUCCESS


 



#
# Use this class to decorate the history with the reference values configured by the user 
#
class Reference( Logger ):

  #
  # Constructor
  #
  def __init__( self , refFile=None, targets=None):
    Logger.__init__(self)
    self.__references = collections.OrderedDict()

    # Set all references from the reference file and target list
    if refFile and targets:
      from saphyra.core import ReferenceReader
      refObj = ReferenceReader().load(refFile)
      for ref in targets:
        pd = (refObj.getSgnPassed(ref[0]) , refObj.getSgnTotal(ref[0]))
        fa = (refObj.getBkgPassed(ref[0]) , refObj.getBkgTotal(ref[0]))
        self.add_reference( ref[0], ref[1], pd, fa )
 

  #
  # Add the reference value
  #
  def add_reference( self, key, reference, pd, fa ):
    pd = [pd[0]/float(pd[1]), pd[0],pd[1]]
    fa = [fa[0]/float(fa[1]), fa[0],fa[1]]
    MSG_INFO( self, '%s | %s(pd=%1.2f, fa=%1.2f, sp=%1.2f)', key, reference, pd[0]*100, fa[0]*100, sp_func(pd[0],fa[0])*100 )
    self.__references[key] = {'pd':pd, 'fa':fa, 'sp':sp_func(pd[0],fa[0]), 'reference' : reference}


  #
  # decorate the history after the training phase
  #
  def decorate( self, history, context ):
    

    model  = context.getHandler("model")
    imodel = context.getHandler("imodel")
    index  = context.getHandler("index")
    sort   = context.getHandler("sort" )
    init   = context.getHandler("init" )

    x_train, y_train = context.getHandler("trnData")
    x_val , y_val    = context.getHandler("valData")

    y_pred = model.predict( x_train, batch_size = 1024, verbose=0 )
    y_pred_val = model.predict( x_val, batch_size = 1024, verbose=0 )

    # get vectors for operation mode (train+val)
    y_pred_operation = np.concatenate( (y_pred, y_pred_val), axis=0)
    y_operation = np.concatenate((y_train,y_val), axis=0)


    train_total = len(y_train)
    val_total = len(y_val)

    # Here, the threshold is variable and the best values will
    # be setted by the max sp value found in hte roc curve
    # Training
    fa, pd, thresholds = roc_curve(y_train, y_pred)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )

    # Validation
    fa_val, pd_val, thresholds_val = roc_curve(y_val, y_pred_val)
    sp_val = np.sqrt(  np.sqrt(pd_val*(1-fa_val)) * (0.5*(pd_val+(1-fa_val)))  )

    # Operation
    fa_op, pd_op, thresholds_op = roc_curve(y_operation, y_pred_operation)
    sp_op = np.sqrt(  np.sqrt(pd_op*(1-fa_op)) * (0.5*(pd_op+(1-fa_op)))  )


    history['reference'] = {}

    for key, ref in self.__references.items():
      d = self.calculate( y_train, y_val , y_operation, ref, pd, fa, sp, thresholds, pd_val, fa_val, sp_val, thresholds_val, pd_op,fa_op,sp_op,thresholds_op )
      MSG_INFO(self, "          : %s", key )
      MSG_INFO(self, "Reference : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", ref['pd'][0]*100, ref['fa'][0]*100, ref['sp']*100 )
      MSG_INFO(self, "Train     : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd'][0]*100, d['fa'][0]*100, d['sp']*100 )
      MSG_INFO(self, "Validation: [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd_val'][0]*100, d['fa_val'][0]*100, d['sp_val']*100 )
      MSG_INFO(self, "Operation : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd_op'][0]*100, d['fa_op'][0]*100, d['sp_op']*100 )
      history['reference'][key] = d




  #
  # Calculate sp, pd and fake given a reference
  # 
  def calculate( self, y_train, y_val , y_op, ref, pd,fa,sp,thresholds, pd_val,fa_val,sp_val,thresholds_val, pd_op,fa_op,sp_op,thresholds_op ):

    d = {}
    def closest( values , ref ):
      index = np.abs(values-ref)
      index = index.argmin()
      return values[index], index


    # Check the reference counts
    op_total = len(y_op[y_op==1])
    if ref['pd'][2] !=  op_total:
      ref['pd'][2] = op_total
      ref['pd'][1] = int(ref['pd'][0]*op_total)

    # Check the reference counts
    op_total = len(y_op[y_op!=1])
    if ref['fa'][2] !=  op_total:
      ref['fa'][2] = op_total
      ref['fa'][1] = int(ref['fa'][0]*op_total)


    d['pd_ref'] = ref['pd']
    d['fa_ref'] = ref['fa']
    d['sp_ref'] = ref['sp']
    d['reference'] = ref['reference']


    # Train
    _, index = closest( pd, ref['pd'][0] )
    train_total = len(y_train[y_train==1])
    d['pd'] = ( pd[index],  int(train_total*float(pd[index])),train_total)
    train_total = len(y_train[y_train!=1])
    d['fa'] = ( fa[index],  int(train_total*float(fa[index])),train_total)
    d['sp'] = sp_func(d['pd'][0], d['fa'][0])
    d['threshold'] = thresholds[index]


    # Validation
    _, index = closest( pd_val, ref['pd'][0] )
    val_total = len(y_val[y_val==1])
    d['pd_val'] = ( pd_val[index],  int(val_total*float(pd_val[index])),val_total)
    val_total = len(y_val[y_val!=1])
    d['fa_val'] = ( fa_val[index],  int(val_total*float(fa_val[index])),val_total)
    d['sp_val'] = sp_func(d['pd_val'][0], d['fa_val'][0])
    d['threshold_val'] = thresholds_val[index]


    # Train + Validation
    _, index = closest( pd_op, ref['pd'][0] )
    op_total = len(y_op[y_op==1])
    d['pd_op'] = ( pd_op[index],  int(op_total*float(pd_op[index])),op_total)
    op_total = len(y_op[y_op!=1])
    d['fa_op'] = ( fa_op[index],  int(op_total*float(fa_op[index])),op_total)
    d['sp_op'] = sp_func(d['pd_op'][0], d['fa_op'][0])
    d['threshold_op'] = thresholds_op[index]

    return d








class Relevance(Logger):

  #
  # Constructor
  #
  def __init__( self , feature_names, method ):
    Logger.__init__(self)
    self.__feature_names = feature_names
    self.__method = method
 


  #
  # decorate the history after the training phase
  #
  def decorate( self, history, context ):
    

    model            = context.getHandler( "model"  )
    x_train, y_train = context.getHandler("trnData" )
    x_val , y_val    = context.getHandler("valData" )
    features         = context.getHandler("features")


    y_pred = model.predict( x_train, batch_size = 1024, verbose=0 )
    y_pred_val = model.predict( x_val, batch_size = 1024, verbose=0 )
    
    # get vectors for operation mode (train+val)
    y_pred_operation = np.concatenate( (y_pred, y_pred_val), axis=0)
    y_operation = np.concatenate((y_train,y_val), axis=0)


    d = {}

    # No threshold is needed
    d['auc_op'] = roc_auc_score(y_operation, y_pred_operation)
    d['mse_op'] = mean_squared_error(y_operation, y_pred_operation)



    # Here, the threshold is variable and the best values will
    # be setted by the max sp value found in hte roc curve
    # Training
    fa, pd, thresholds = roc_curve(y_operation, y_pred_operation)
    sp = ( np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  ) )
    knee = np.argmax(sp) 
    threshold = thresholds[knee]
   
    # Hold some benchmarks for relevance analysis
    d['sp_op'] = sp[knee]

 


    h = {
         'before'  : d,
         'foreach' : [],
        }

    MSG_INFO( self, "Calculate the relevance for each selected feature..." )
    for name in self.__feature_names:
     
      d_name = {}
      d_name['feature'] = name

      MSG_INFO( self, "Deactivate feature with name: %s", name )
      y_pred = self.__deactivate_feature_and_predict( model, x_train, features, name , self.__method)
      y_pred_val = self.__deactivate_feature_and_predict( model, x_val, features, name, self.__method)

      # get vectors for operation mode (train+val)
      y_pred_operation = np.concatenate( (y_pred, y_pred_val), axis=0)
      y_operation = np.concatenate((y_train,y_val), axis=0)

      d_name['auc_op'] = roc_auc_score(y_operation, y_pred_operation)
      d_name['mse_op'] = mean_squared_error(y_operation, y_pred_operation)


      # Here, the threshold is variable and the best values will
      # be setted by the max sp value found in hte roc curve
      # Training
      fa, pd, thresholds = roc_curve(y_train, y_pred)
      sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
      knee = np.argmax(sp)
      max_sp = sp[knee]
      threshold = thresholds[knee]

      d_name['sp_op'] = sp[knee]
      
      h['foreach'].append( d_name )

      value = (d['sp_op']-d_name['sp_op'])*100
      status = 'Confusion' if value < 0 else 'Relevant'

      MSG_INFO( self,  "Relevance (%s), deltaSP: %1.2f (%s)", name, value, status )

    
    history['relevance'] = h



  def __deactivate_feature_and_predict( self, model, data, all_features, feature, method='by_mean' ):

    # locate the feature position given by name
    input_idx, feature_idx = self.__where( all_features , feature )

    local_data = copy(data)
    if method=='by_mean':
      local_data[input_idx][:,feature_idx] = np.mean( local_data[input_idx][:,feature_idx] )
      return model.predict( local_data, batch_size=1024, verbose=0)
    else:
      MSG_FATAL( self, "Deactivation method not reconized: %s", method )



  def __where( self, all_features, wanted_feature ):
    # where is it first?
    for idx, features_for_this_input in enumerate(all_features):
      for jdx, feature in enumerate(features_for_this_input):
        if feature==wanted_feature:
          return idx, jdx
    









