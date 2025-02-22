
__all__ = ["create_jobs"]

from Gaugi import declareProperty, mkdir_p
from Gaugi import Logger
from Gaugi.macros import *

from sklearn.model_selection import KFold

# A simple solution need to refine the documentation
from itertools import product
def create_iter(fun, n_items_per_job, items_lim):
  return ([fun(i, n_items_per_job)
           if (i+n_items_per_job) <= items_lim 
           else fun(i, items_lim % n_items_per_job) 
           for i in range(0, items_lim, n_items_per_job)])


# default model (ringer vanilla)
# Remove the keras dependence and get keras from tensorflow 2.0
import tensorflow as tf
default_model = tf.keras.Sequential()
default_model.add(tf.keras.layers.Dense(5, input_shape=(100,), activation='tanh', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
default_model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
default_model.add(tf.keras.layers.Activation('tanh'))
 


class create_configuration_jobs( Logger ):
  '''
  Documentation (TODO)
  '''

  def __init__( self, **kw):

    Logger.__init__(self, **kw)

  def time_stamp(self):
    from datetime import datetime
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H.%M.%S")
    return timestampStr

  def __call__( self, **kw):     
    # Cross validation configuration
    declareProperty( self, kw, 'outputFolder' ,       'jobConfig'           )
    declareProperty( self, kw, 'sortBounds'   ,             5               )
    declareProperty( self, kw, 'nInits'       ,             10              )
    declareProperty( self, kw, 'nSortsPerJob' ,             1               )
    declareProperty( self, kw, 'nInitsPerJob' ,             10              ) 
    declareProperty( self, kw, 'nModelsPerJob',             1               ) 
    declareProperty( self, kw, 'models'       ,   [default_model]           )
    declareProperty( self, kw, 'model_tags'   ,   ['mlp_100_5_1']           )
    declareProperty( self, kw, 'crossval'     , KFold(10,shuffle=True, random_state=512)  )

    time_stamp = self.time_stamp()    
    # creating the job mechanism file first
    mkdir_p(self.outputFolder)

    if type(self.models) is not list:
      self.models = [self.models]
    
    modelJobsWindowList = create_iter(lambda i, sorts: list(range(i, i+sorts)), 
                                      self.nModelsPerJob,
                                      len(self.models))
    sortJobsWindowList  = create_iter(lambda i, sorts: list(range(i, i+sorts)), 
                                      self.nSortsPerJob,
                                      self.sortBounds)
    initJobsWindowList  = create_iter(lambda i, sorts: list(range(i, i+sorts)), 
                                      self.nInitsPerJob, 
                                      self.nInits)

    nJobs = 0 
    for (model_idx_list, sort_list, init_list) in product(modelJobsWindowList,
                                                          sortJobsWindowList, 
                                                          initJobsWindowList):

      MSG_INFO( self,
                'Creating job config with sort (%d to %d) and %d inits and model Index %d to %d', 
                sort_list[0], sort_list[-1], len(init_list), model_idx_list[0], model_idx_list[-1] )

      from saphyra.core.readers.versions import Job_v1
      job = Job_v1()
      # to be user by the database table
      job.setId( nJobs )
      job.setSorts(sort_list)
      job.setInits(init_list)
      job.setModels([self.models[idx] for idx in model_idx_list],  model_idx_list )
      # save config file
      model_str = 'ml%i.mu%i' %(model_idx_list[0], model_idx_list[-1])
      sort_str  = 'sl%i.su%i' %(sort_list[0], sort_list[-1])
      init_str  = 'il%i.iu%i' %(init_list[0], init_list[-1])
      job.save( self.outputFolder+'/' + ('job_config.ID_%s.%s_%s_%s') %
              ( str(nJobs).zfill(4), model_str, sort_str, init_str) )
      nJobs+=1

    MSG_INFO( self, "A total of %d jobs...", nJobs)

create_jobs = create_configuration_jobs()


