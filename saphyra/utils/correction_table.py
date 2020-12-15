
from Gaugi.messenger import Logger
from Gaugi.messenger.macros import *
from Gaugi.monet.AtlasStyle import *
from Gaugi.monet.PlotFunctions import *
from Gaugi.monet.TAxisFunctions import *

from ROOT import kBlack,kBlue,kRed,kAzure,kGreen,kMagenta,kCyan,kOrange,kGray,kYellow,kWhite
from Gaugi.monet import *
from array import array
from copy import deepcopy
import time,os,math,sys,pprint,glob,warnings
import numpy as np
import pandas as pd
import ROOT, math
import ctypes
import collections


#
# Correction class table
#
class correction_table(Logger):

    #
    # Constructor
    #
    def __init__(self, generator, etbins, etabins, x_bin_size, y_bin_size, ymin, ymax, false_alarm_limit=0.5):
        
        Logger.__init__(self)
        self.__generator = generator
        self.__etbins = etbins
        self.__etabins = etabins
        self.__ymin = ymin
        self.__ymax = ymax
        self.__x_bin_size = x_bin_size
        self.__y_bin_size = y_bin_size
        self.__false_alarm_limit = false_alarm_limit



    #
    # Fill correction table 
    #
    def fill( self, data_paths,  models ):


        # make template dataframe
        dataframe = collections.OrderedDict({
                      'name':[],
                      'reference_passed':[],
                      'reference_total':[],
                      'reference_eff':[],
                      'et_bin':[],
                      'eta_bin':[],
                      'signal_passed':[],
                      'signal_total':[],
                      'signal_eff':[],
                      'background_passed':[],
                      'background_total':[],
                      'background_eff':[],
                      'signal_corrected_passed':[],
                      'signal_corrected_total':[],
                      'signal_corrected_eff':[],
                      'background_corrected_passed':[],
                      'background_corrected_total':[],
                      'background_corrected_eff':[],
                      'th2_signal':[],
                      'th2_background':[]
                     })

        # reduce verbose
        def add(key,value):
          dataframe[key].append(value)


        # Loop over all et/eta bins
        for et_bin in range(len(self.__etbins)-1):
            for eta_bin in range(len(self.__etabins)-1):

                path = data_paths[et_bin][eta_bin]
                data, target, avgmu, references = self.__generator(path)
                model = models[et_bin][eta_bin]
                model['thresholds'] = {}

                # Get the predictions
                outputs = model['model'].predict(data, batch_size=1024, verbose=1).flatten()

                # Get all limits using the output
                xmin = int(np.percentile(outputs , 1))
                xmax = int(np.percentile(outputs, 99))
                MSG_INFO(self, 'Setting xmin to %1.2f and xmax to %1.2f', xmin, xmax)
                xbins = int((xmax-xmin)/self.__x_bin_size)
                ybins = int((self.__ymax-self.__ymin)/self.__y_bin_size)
                
                # Fill 2D histograms
                from ROOT import TH2F
                import array
                th2_signal = TH2F( 'th2_signal_et%d_eta%d'%(et_bin,eta_bin), '', xbins, xmin, xmax, ybins, self.__ymin, self.__ymax )
                w = array.array( 'd', np.ones_like( outputs[target==1] ) )
                th2_signal.FillN( len(outputs[target==1]), array.array('d',  outputs[target==1].tolist()),  array.array('d',avgmu[target==1].tolist()), w)
                th2_background = TH2F( 'th2_background_et%d_eta%d'%(et_bin,eta_bin), '', xbins,xmin, xmax, ybins, self.__ymin, self.__ymax )
                w = array.array( 'd', np.ones_like( outputs[target==0] ) )
                th2_background.FillN( len(outputs[target==0]), array.array('d',outputs[target==0].tolist()), array.array('d',avgmu[target==0].tolist()), w)

                MSG_INFO( self, 'Apply correction to: et%d_eta%d', et_bin, eta_bin)

                for name, ref in references.items():

                    reference_num = ref['signal_passed']
                    reference_den = ref['signal_total']
                    target = reference_num/reference_den


                    false_alarm = 1.0
                    while false_alarm > self.__false_alarm_limit:

                        threshold, _ = self.find_threshold( th2_signal.ProjectionX(), target )
                        # Get the efficiency without linear adjustment
                        signal_noadjustment_eff, signal_noadjustment_num, signal_noadjustment_den = \
                                                                self.calculate_num_and_den(th2_signal, 0.0, threshold)
                        background_noadjustment_eff, background_noadjustment_num, background_noadjustment_den = \
                                                                self.calculate_num_and_den(th2_background, 0.0, threshold)

                        # Apply the linear adjustment and fix it in case of positive slope
                        slope, offset = self.fit( th2_signal, target )
                        slope = 0 if slope>0 else slope
                        offset = threshold if slope>0 else offset
                        if slope>0:
                          MSG_WARNING(self, "Retrieved positive angular factor of the linear correction, setting to 0!")

                        # Get the efficiency with linear adjustment
                        signal_eff, signal_num, signal_den = self.calculate_num_and_den(th2_signal, slope, offset)
                        background_eff, background_num, background_den = self.calculate_num_and_den(th2_background, slope, offset)

                        false_alarm = background_num/background_den # get the passed/total

                        if false_alarm > self.__false_alarm_limit:
                            # Reduce the reference value by hand
                            value-=0.0025

                    MSG_INFO( self, 'Reference name: %s, target: %1.2f%%', name, target*100 )
                    MSG_INFO( self, 'Signal with correction is: %1.2f%%', signal_num/signal_den * 100 )
                    MSG_INFO( self, 'Background with correction is: %1.2f%%', background_num/background_den * 100 )

                    # decore the model array
                    model['thresholds'][name] = {'offset':offset, 'slope':slope, 'offset_noadjust' : threshold}

                    # Save some values into the main table
                    add( 'name'                        , name )
                    add( 'et_bin'                      , et_bin  )
                    add( 'eta_bin'                     , eta_bin )
                    add( 'reference_passed'            , reference_num )
                    add( 'reference_total'             , reference_den )
                    add( 'reference_eff'               , reference_num/reference_den )
                    add( 'signal_passed'               , signal_num )
                    add( 'signal_total'                , signal_den )
                    add( 'signal_eff'                  , signal_num/signal_den )
                    add( 'background_passed'           , background_num )
                    add( 'background_total'            , background_den )
                    add( 'background_eff'              , background_num/background_den )
                    #add( 'signal_corrected_passed'     , signal_corrected_num )
                    #add( 'signal_corrected_total'      , signal_corrected_den )
                    #add( 'signal_corrected_eff'        , signal_corrected_num/signal_corrected_den )
                    #add( 'background_corrected_passed' , background_corrected_num )
                    #add( 'background_corrected_total'  , background_corrected_den )
                    #add( 'background_corrected_eff'    , background_corrected_num/background_corrected_den )
                    add( 'th2_signal'                  , th2_signal )
                    add( 'th2_background'              , th2_background )

        
        self.__table = pd.Dataframe( dataframe )
        self.__table.head()

   

    


    #
    # Export all models ringer
    #
    def export( self, models, model_output_format , conf_output, reference_name, to_onnx=False):


        from ROOT import TEnv

        model_etmin_vec = []
        model_etmax_vec = []
        model_etamin_vec = []
        model_etamax_vec = []
        model_paths = []

        slopes = []
        offsets = []

        # serialize all models
        for model in models:

            model_etmin_vec.append( model['etBin'][0] )
            model_etmax_vec.append( model['etBin'][1] )
            model_etamin_vec.append( model['etaBin'][0] )
            model_etamax_vec.append( model['etaBin'][1] )

            etBinIdx = model['etBinIdx']
            etaBinIdx = model['etaBinIdx']

            model_name = model_output_format%( etBinIdx, etaBinIdx )
            model_paths.append( model_name )

            # Save onnx mode!
            if to_onnx:
                import onnx, keras2onnx
                onnx_model = keras2onnx.convert_keras(model['model'], model['model'].name)
                onnx.save_model(onnx_model, model_name+'.onnx')

            model_json = model['model'].to_json()
            with open(model_name+".json", "w") as json_file:
                json_file.write(model_json)
                # saving the model weight separately
                model['model'].save_weights(model_name+".h5")

            slopes.append( model['thresholds'][reference_name]['slope'] )
            offsets.append( model['thresholds'][reference_name]['offsets'] )


        def list_to_str( l ):
            s = str()
            for ll in l:
              s+=str(ll)+'; '
            return s[:-2]

        # Write the config file
        file = TEnv( 'ringer' )
        file.SetValue( "__name__", 'should_be_filled' )
        file.SetValue( "__version__", 'should_be_filled' )
        file.SetValue( "__operation__", reference_name )
        file.SetValue( "__signature__", 'should_be_filled' )
        file.SetValue( "Model__size"  , str(len(models)) )
        file.SetValue( "Model__etmin" , list_to_str(model_etmin_vec) )
        file.SetValue( "Model__etmax" , list_to_str(model_etmax_vec) )
        file.SetValue( "Model__etamin", list_to_str(model_etamin_vec) )
        file.SetValue( "Model__etamax", list_to_str(model_etamax_vec) )
        file.SetValue( "Model__path"  , list_to_str( model_paths ) )
        file.SetValue( "Threshold__size"  , str(len(models)) )
        file.SetValue( "Threshold__etmin" , list_to_str(model_etmin_vec) )
        file.SetValue( "Threshold__etmax" , list_to_str(model_etmax_vec) )
        file.SetValue( "Threshold__etamin", list_to_str(model_etamin_vec) )
        file.SetValue( "Threshold__etamax", list_to_str(model_etamax_vec) )
        file.SetValue( "Threshold__slope" , list_to_str(slopes) )
        file.SetValue( "Threshold__offset", list_to_str(offsets) )
        file.SetValue( "Threshold__MaxAverageMu", 100)
        file.WriteFile(conf_output)




    #
    # Find the threshold given a reference value
    #
    def find_threshold(self, th1,effref):
        nbins = th1.GetNbinsX()
        fullArea = th1.Integral(0,nbins+1)
        if fullArea == 0:
            return 0,1
        notDetected = 0.0; i = 0
        while (1. - notDetected > effref):
            cutArea = th1.Integral(0,i)
            i+=1
            prevNotDetected = notDetected
            notDetected = cutArea/fullArea
        eff = 1. - notDetected
        prevEff = 1. -prevNotDetected
        deltaEff = (eff - prevEff)
        threshold = th1.GetBinCenter(i-1)+(effref-prevEff)/deltaEff*(th1.GetBinCenter(i)-th1.GetBinCenter(i-1))
        error = 1./math.sqrt(fullArea)
        return threshold, error

    #
    # Get all points in the 2D histogram given a reference value
    #
    def get_points( self, th2 , effref):
        nbinsy = th2.GetNbinsY()
        x = list(); y = list(); errors = list()
        for by in range(nbinsy):
            xproj = th2.ProjectionX('xproj'+str(time.time()),by+1,by+1)
            discr, error = self.find_threshold(xproj,effref)
            dbin = xproj.FindBin(discr)
            x.append(discr); y.append(th2.GetYaxis().GetBinCenter(by+1))
            errors.append( error )
        return x,y,errors



    #
    # Calculate the linear fit given a 2D histogram and reference value and return the slope and offset
    #
    def fit(self, th2,effref):
        x_points, y_points, error_points = self.get_points(th2, effref )
        import array
        g = ROOT.TGraphErrors( len(x_points)
                             , array.array('d',y_points,)
                             , array.array('d',x_points)
                             , array.array('d',[0.]*len(x_points))
                             , array.array('d',error_points) )
        firstBinVal = th2.GetYaxis().GetBinLowEdge(th2.GetYaxis().GetFirst())
        lastBinVal = th2.GetYaxis().GetBinLowEdge(th2.GetYaxis().GetLast()+1)
        f1 = ROOT.TF1('f1','pol1',firstBinVal, lastBinVal)
        g.Fit(f1,"FRq")
        slope = f1.GetParameter(1)
        offset = f1.GetParameter(0)
        return slope, offset


    #
    # Calculate the numerator and denomitator given the 2D histogram and slope/offset parameters
    #
    def calculate_num_and_den(self, th2, slope, offset) :

      nbinsy = th2.GetNbinsY()
      th1_num = th2.ProjectionY(th2.GetName()+'_proj'+str(time.time()),1,1)
      th1_num.Reset("ICESM")
      numerator=0; denominator=0
      # Calculate how many events passed by the threshold
      for by in range(nbinsy) :
          xproj = th2.ProjectionX('xproj'+str(time.time()),by+1,by+1)
          # Apply the correction using ax+b formula
          threshold = slope*th2.GetYaxis().GetBinCenter(by+1)+ offset
          dbin = xproj.FindBin(threshold)
          num = xproj.Integral(dbin+1,xproj.GetNbinsX()+1)
          th1_num.SetBinContent(by+1,num)
          numerator+=num
          denominator+=xproj.Integral(-1, xproj.GetNbinsX()+1)

      # Calculate the efficiency histogram
      th1_den = th2.ProjectionY(th2.GetName()+'_proj'+str(time.time()),1,1)
      th1_eff = th1_num.Clone()
      th1_eff.Divide(th1_den)
      # Fix the error bar
      for bx in range(th1_eff.GetNbinsX()):
          if th1_den.GetBinContent(bx+1) != 0 :
              eff = th1_eff.GetBinContent(bx+1)
              try:
                  error = math.sqrt(eff*(1-eff)/th1_den.GetBinContent(bx+1))
              except:
                  error=0
              th1_eff.SetBinError(bx+1,eff)
          else:
              th1_eff.SetBinError(bx+1,0)

      return th1_eff, numerator, denominator







if __name__ == "__main__":

    from saphyra.utils import crossval_table

    def create_op_dict(op):
        d = {
                  op+'_pd_ref'    : "reference/"+op+"_cutbased/pd_ref#0",
                  op+'_fa_ref'    : "reference/"+op+"_cutbased/fa_ref#0",
                  op+'_sp_ref'    : "reference/"+op+"_cutbased/sp_ref",
                  op+'_pd_val'    : "reference/"+op+"_cutbased/pd_val#0",
                  op+'_fa_val'    : "reference/"+op+"_cutbased/fa_val#0",
                  op+'_sp_val'    : "reference/"+op+"_cutbased/sp_val",
                  op+'_pd_op'     : "reference/"+op+"_cutbased/pd_op#0",
                  op+'_fa_op'     : "reference/"+op+"_cutbased/fa_op#0",
                  op+'_sp_op'     : "reference/"+op+"_cutbased/sp_op",
                
                  # Counts
                  op+'_pd_ref_passed'    : "reference/"+op+"_cutbased/pd_ref#1",
                  op+'_fa_ref_passed'    : "reference/"+op+"_cutbased/fa_ref#1",
                  op+'_pd_ref_total'     : "reference/"+op+"_cutbased/pd_ref#2",
                  op+'_fa_ref_total'     : "reference/"+op+"_cutbased/fa_ref#2",   
                  op+'_pd_val_passed'    : "reference/"+op+"_cutbased/pd_val#1",
                  op+'_fa_val_passed'    : "reference/"+op+"_cutbased/fa_val#1",
                  op+'_pd_val_total'     : "reference/"+op+"_cutbased/pd_val#2",
                  op+'_fa_val_total'     : "reference/"+op+"_cutbased/fa_val#2",  
                  op+'_pd_op_passed'     : "reference/"+op+"_cutbased/pd_op#1",
                  op+'_fa_op_passed'     : "reference/"+op+"_cutbased/fa_op#1",
                  op+'_pd_op_total'      : "reference/"+op+"_cutbased/pd_op#2",
                  op+'_fa_op_total'      : "reference/"+op+"_cutbased/fa_op#2",
        } 
        return d
    
    
    tuned_info = collections.OrderedDict( {
                  # validation
                  "max_sp_val"      : 'summary/max_sp_val',
                  "max_sp_pd_val"   : 'summary/max_sp_pd_val#0',
                  "max_sp_fa_val"   : 'summary/max_sp_fa_val#0',
                  # Operation
                  "max_sp_op"       : 'summary/max_sp_op',
                  "max_sp_pd_op"    : 'summary/max_sp_pd_op#0',
                  "max_sp_fa_op"    : 'summary/max_sp_fa_op#0',
                  } )
    
    tuned_info.update(create_op_dict('tight'))
    tuned_info.update(create_op_dict('medium'))
    tuned_info.update(create_op_dict('loose'))
    tuned_info.update(create_op_dict('vloose'))
    
    
    etbins = [15,20,30,40,50,100000]
    etabins = [0, 0.8 , 1.37, 1.54, 2.37, 2.5]

    cv  = crossval_table( tuned_info, etbins = etbins , etabins = etabins )
    #cv.fill( '/home/jodafons/public/tunings/v10/*.r2/*/*.gz', 'v10')
    #cv.to_csv( 'v10.csv' )
    cv.from_csv( 'v10.csv' )
    best_inits = cv.filter_inits("max_sp_val")
    best_inits = best_inits.loc[(best_inits.model_idx==0)]
    best_sorts = cv.filter_sorts(best_inits, 'max_sp_val')
    best_models = cv.get_best_models(best_sorts, remove_last=True)


    
    #
    # Generator to read, prepare data and get all references
    #
    def generator( path ):

        def norm1( data ):
            norms = np.abs( data.sum(axis=1) )
            norms[norms==0] = 1
            return data/norms[:,None]
        from Gaugi import load
        import numpy as np
        d = load(path)
        feature_names = d['features'].tolist()
        data = norm1(d['data'][:,1:101])
        target = d['target']
        avgmu = d['data'][:,0]
        references = ['T0HLTElectronT2CaloTight','T0HLTElectronT2CaloMedium','T0HLTElectronT2CaloLoose','T0HLTElectronT2CaloVLoose']
        ref_dict = {}
        for ref in references:
            answers = d['data'][:, feature_names.index(ref)]
            signal_passed = sum(answers[target==1])
            signal_total = len(answers[target==1])
            background_passed = sum(answers[target==0])
            background_total = sum(answers[target==0])
            pd = signal_passed/signal_total
            fa = background_passed/background_total
            ref_dict[ref] = {'signal_passed': signal_passed, 'signal_total': signal_total, 'pd' : pd,
                             'background_passed': background_passed, 'background_total': background_total, 'fa': fa}

        return data, target, avgmu, ref_dict


    
    path = '~/public/cern_data/files/Zee/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et{ET}_eta{ETA}.npz'

    paths = [[ path.format(ET=et,ETA=eta) for eta in range(5)] for et in range(5)]

    # get best models

    ct  = correction_table( generator, etbins , etabins, 0.02, 0.5, 16, 70 )
    ct.fill(paths, best_models)





