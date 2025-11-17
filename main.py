# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings


gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES']=gpu
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow CUDA load statements
#warnings.filterwarnings('ignore')

from keras import backend as K
import tensorflow as tf

import copy
import argparse
import numpy as np


gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", gpus)

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    tf.config.experimental.set_memory_growth(gpus[int(gpu)], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


import utilArgparse
import utilConst
import utilIO
import util
import CNNmodel

#util.init()

K.set_image_data_format('channels_last')




# ----------------------------------------------------------------------------
def menu():
    parser = argparse.ArgumentParser(description='Binarization with masking layer and oversampling')

    
    parser.add_argument('-db_train_src', required=True, help='Dataset path for training (src imags)')
    parser.add_argument('-db_train_gt', required=True, help='Dataset path for training (gt images)')

    parser.add_argument('-db_test_src', required=False, help='Dataset path to test (src imags)')
    parser.add_argument('-db_test_gt', required=False, help='Dataset path to test (gt images)')

    parser.add_argument('-aug',   nargs='*',
                        choices=utilConst.AUGMENTATION_CHOICES,
                        default=[utilConst.AUGMENTATION_NONE], 
                        help='Data augmentation modes')

    parser.add_argument('-npatches', default=-1, dest='n_pa', type=int,   help='Number of patches to be extracted from training data')
    
    parser.add_argument('-n_annotated_patches', default=-1, dest='n_an', type=int,   help='Number of patches to be extracted from training data')

    parser.add_argument('-window_w', default=256, dest='win_w', type=int,   help='width of window')
    parser.add_argument('-window_h', default=256, dest='win_h', type=int,   help='height of window')

    parser.add_argument('-l',          default=4,        dest='n_la',     type=int,   help='Number of layers')
    parser.add_argument('-f',          default=64,      dest='nb_fil',   type=int,   help='Number of filters')
    parser.add_argument('-k',          default=5,        dest='ker',            type=int,   help='kernel size')
    parser.add_argument('-drop',   default=0.2,        dest='drop',          type=float, help='dropout value')
    
    parser.add_argument('-ink_rate',   default=0.025,        dest='ink_rate',          type=float, help='Ink proportion to select patches to be annotated')
    

    parser.add_argument('-pages_train',   default=-1,      type=int,   help='Number of pages to be used for training. -1 to load all the training set.')

    parser.add_argument('-e',           default=200,    dest='ep',            type=int,   help='nb_epoch')
    parser.add_argument('-b',           default=16,     dest='ba',               type=int,   help='batch size')
    parser.add_argument('-verbose',     default=1,                                  type=int,   help='1=show batch increment, other=mute')

    parser.add_argument('--test',   action='store_true', help='Only run test')
    parser.add_argument('--all_ths',   action='store_true', help='Run the test evaluating all the possible binarization thresholds on the probabilistic map obtained by the model.')
    
    
    parser.add_argument('-res', required=False, help='File where append the results.')
    parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')
    parser.add_argument('-no_mask', required=False, action='store_true', help='File where append the results.')
    
    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args

def tpc_result(result):
    return round(result*100,1)
  
def number_to_string(number):
    return str(tpc_result(number)).replace(".",",")


def evaluate_with_binarization_threshold(
                  number_annotated_patches,
                  number_annotated_patches_val,
                  path_model, 
                  val_data, 
                  test_data,
                  config, 
                  input_shape, 
                  threshold):
    if number_annotated_patches > 0 and number_annotated_patches_val > 0:
          print("Results of the test...")
          best_fm_val, best_th_val, prec_val, recall_val, iou_val, specificity_val, tp_val, tn_val, fp_val, fn_val, dict_predictions = util.compute_best_threshold(path_model, val_data, config.ba, input_shape, config.ink_rate, nb_annotated_patches=config.n_an, threshold=threshold, with_masked_input=False)
          with_mask = not config.no_mask
          dict_results = util.test_model(config, path_model, test_data, input_shape, best_th_val, with_mask)
          # dict_results: best_fm, prec, recall, iou, specificity, tp, tn, fp, fn
          best_fm_test = dict_results[utilConst.KEY_RESULT][0][0]
          prec_test = dict_results[utilConst.KEY_RESULT][0][1]
          recall_test = dict_results[utilConst.KEY_RESULT][0][2]
          iou_test = dict_results[utilConst.KEY_RESULT][0][3]
          specificity_test = dict_results[utilConst.KEY_RESULT][0][4]
          tp_test = dict_results[utilConst.KEY_RESULT][0][5]
          tn_test = dict_results[utilConst.KEY_RESULT][0][6]
          fp_test = dict_results[utilConst.KEY_RESULT][0][7]
          fn_test = dict_results[utilConst.KEY_RESULT][0][8]
          avg_elapsed = dict_results[utilConst.KEY_RESULT][0][9]
    else:
          print("No model is trained with this configuration...")
          best_fm_val = 0
          best_th_val = 0
          prec_val = 0
          recall_val = 0
          dict_predictions = None
          best_fm_test = 0
          prec_test = 0
          recall_test = 0
          iou_test = 0
          specificity_test = 0
          tp_test = 0
          tn_test = 0
          fp_test = 0
          fn_test = 0
          avg_elapsed = 0
        
    separator = ";"
    print ("SUMMARY:")
    str_header = "Train" + separator
    str_header += "Test" + separator
    str_header += "PAG" + separator
    str_header += "Num pages train" + separator
    str_header += "ANN" + separator
    str_header += "Num annotations per page" + separator
    str_header += "PAT" + separator
    str_header += "Num random patches" + separator
    str_header += "Ink rate" + separator
    str_header += "VAL" + separator
    str_header += "Th_bin" + separator
    str_header += "F1-val" + separator
    str_header += "P-val" + separator
    str_header += "R-val" + separator
    str_header += "Num annotated patches-val" + separator
    str_header += "Maximum num annotated patches-val" + separator
    str_header += "TEST" + separator
    str_header += "F1-test" + separator
    str_header += "P-test" + separator
    str_header += "R-test" + separator
    str_header += "Num annotated patches-test" + separator
    str_header += "Maximum num annotated patches-test" + separator
    str_header += "IoU-test" + separator
    str_header += "Specificity-test" + separator
    str_header += "TP-test" + separator
    str_header += "TN-test" + separator
    str_header += "FP-test" + separator
    str_header += "FN-test" + separator
    str_header += "TimePerPage(s)"+ separator

    str_properties = str(config.db_train_src) + separator
    str_properties += str(config.db_test_src) + separator
    str_properties += "PAG" + separator
    str_properties += str(config.pages_train) + separator
    str_properties += "ANN" + separator
    str_properties += str(config.n_an) + separator
    str_properties += "PAT" + separator
    str_properties += str(config.n_pa) + separator
    str_properties += str(config.ink_rate).replace(".",",") + separator  
    str_result = str_properties+separator
    str_result += "VAL"+separator
    str_result += str(best_th_val).replace(".",",") + separator
    str_result += number_to_string(best_fm_val) + separator
    str_result += number_to_string(prec_val) + separator
    str_result += number_to_string(recall_val) + separator  #number_to_string(best_fm_val) + separator + number_to_string(prec_val) + separator + number_to_string(recall_val) + separator + str(best_th_val).replace(".", ",") + separator
    str_result += str(number_annotated_patches_val) + separator
    str_result += str(max_number_annotated_patches_val) + separator

    print("Results: " + number_to_string(best_fm_test) + separator + number_to_string(prec_test) + separator + number_to_string(recall_test) + separator + number_to_string(iou_test) + separator + number_to_string(specificity_test) + separator + number_to_string(tp_test) + separator + number_to_string(tn_test) + separator + number_to_string(fp_test) + separator + number_to_string(fn_test))
    
    str_result += separator + "TEST" + separator
    str_result += number_to_string(best_fm_test) + separator 
    str_result += number_to_string(prec_test) + separator 
    str_result += number_to_string(recall_test) + separator
    str_result += str(number_annotated_patches) + separator
    str_result += str(max_number_annotated_patches) + separator
    str_result += number_to_string(iou_test) + separator
    str_result += number_to_string(specificity_test) + separator
    str_result += number_to_string(tp_test) + separator
    str_result += number_to_string(tn_test) + separator
    str_result += number_to_string(fp_test) + separator
    str_result += number_to_string(fn_test) + separator
    str_result += str(avg_elapsed).replace(".",",") + separator
      
    if config.res is not None:
        utilIO.appendString(str_result, config.res, True)
    return str_header, str_result


if __name__ == "__main__":
    config = menu()
    print (config)
    
    path_model = utilIO.getPathModel(config)
    utilIO.createParentDirectory(path_model)
    
    input_shape = util.getInputShape(config)
    
    list_src_train = utilIO.listFilesRecursive(config.db_train_src)
    list_gt_train = utilIO.listFilesRecursive(config.db_train_gt)
    assert(len(list_src_train) == len(list_gt_train))

    train_data, val_data = util.create_Validation_and_Training_partitions(
                                        list_src_train=list_src_train, 
                                        list_gt_train=list_gt_train, 
                                        pages_train=config.pages_train)
    
    if config.test == False: # TRAINING MODE

      print("Training and validation partitioned...")
      print("\tTraining: %d" %(len(train_data)))
      print("\tValidation: %d" %(len(val_data)))

      augmentation_val = ["none"]
      if utilConst.AUGMENTATION_RANDOM in config.aug:
        augmentation_val = ["random"]
      
      model = CNNmodel.get_model(input_shape, config.no_mask, config.n_la, config.nb_fil, config.ker, dropout=config.drop, stride=2)
      
      train_generator = util.create_generator(train_data, config.no_mask, config.ba, input_shape, config.n_pa, config.n_an, config.aug, config.ink_rate)
      val_generator = util.create_generator(val_data, config.no_mask, config.ba, input_shape, config.n_pa, config.n_an, augmentation_val, config.ink_rate)
      
      nb_train_pages = len(train_data)
      nb_val_pages = len(val_data)
      
      epochs = config.ep
      patience = 50
      
      print("Number of effective epochs: " + str(epochs))
      print("Effective patience: " + str(patience))

      number_annotated_patches = util.get_number_annotated_patches(train_data, input_shape[0], input_shape[1], config.ink_rate, config.n_pa)  
      number_annotated_patches_val = util.get_number_annotated_patches(val_data, input_shape[0], input_shape[1], config.ink_rate, config.n_pa)  
      
      if utilConst.AUGMENTATION_RANDOM in config.aug:
        assert(config.n_pa!=-1)
        steps_per_epoch = int(np.ceil((config.n_pa*nb_train_pages)/ config.ba))
      else:
        
        print ("Number of annotated patches: " + str(number_annotated_patches))
        steps_per_epoch = np.ceil(number_annotated_patches/config.ba)

      if number_annotated_patches > 0 and number_annotated_patches_val > 0:
        steps_per_epoch = max(1, steps_per_epoch)
        CNNmodel.train(model, path_model, train_generator, val_generator, steps_per_epoch, nb_val_pages, config.ba, epochs, patience=patience)
      else:
        print("No samples available with the ink rate considered. Train (" + str(number_annotated_patches) +") ; Val (" + str(number_annotated_patches_val) + ")")
        
        
    else: #TEST MODE
      
      util.extract_annotated_samples_and_region_mask(path_model, train_data, input_shape, config.ink_rate, config.n_an, with_masked_input=True)
      
      list_src_test = utilIO.listFilesRecursive(config.db_test_src)
      list_gt_test = utilIO.listFilesRecursive(config.db_test_gt)
      assert(len(list_src_test) == len(list_gt_test))
      
      test_data = utilIO.match_SRC_GT_Images(list_src_test, list_gt_test)
      
      number_annotated_patches = util.get_number_annotated_patches(train_data, input_shape[0], input_shape[1], config.ink_rate, config.n_an) 
      number_annotated_patches_val = util.get_number_annotated_patches(val_data, input_shape[0], input_shape[1], config.ink_rate, config.n_an)  

      max_number_annotated_patches = util.get_number_annotated_patches(train_data, input_shape[0], input_shape[1], config.ink_rate, -1) 
      max_number_annotated_patches_val = util.get_number_annotated_patches(val_data, input_shape[0], input_shape[1], config.ink_rate, -1)
      
      print("Obtaining best threshold...(Validation partition)")
      
      threshold=None
      
      if config.all_ths is False:
          str_header,str_result = evaluate_with_binarization_threshold(
                  number_annotated_patches=number_annotated_patches,
                  number_annotated_patches_val=number_annotated_patches_val,
                  path_model=path_model, 
                  val_data=val_data, 
                  test_data=test_data,
                  config=config, 
                  input_shape=input_shape, 
                  threshold=threshold)
      else:
          str_result_complete = ""
          for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            str_header,str_result = evaluate_with_binarization_threshold(
                    number_annotated_patches=number_annotated_patches,
                    number_annotated_patches_val=number_annotated_patches_val,
                    path_model=path_model, 
                    val_data=val_data, 
                    test_data=test_data,
                    config=config, 
                    input_shape=input_shape, 
                    threshold=th)
            str_result_complete += str_result
        
          str_result = str_result_complete

      print ('*'*80)
      print(str_header)
      print(str_result)
      
