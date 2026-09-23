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

'''
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
'''

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
    
    
    parser.add_argument('-res', required=False, help='File where append the results.')
    parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')
    parser.add_argument('-no_mask', required=False, action='store_true', help='File where append the results.')
    
    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args

def tpc_result(result):
    return round(result*100,1)
  
def number_to_string(number, prec=1):
    return str(round(number, prec)).replace(".",",")

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
  
    list_src_test = utilIO.listFilesRecursive(config.db_test_src)
    list_gt_test = utilIO.listFilesRecursive(config.db_test_gt)
    assert(len(list_src_test) == len(list_gt_test))
    test_data = utilIO.match_SRC_GT_Images(list_src_test, list_gt_test)
    
    print("Training and validation partitioned...")
    print("\tTraining: %d" %(len(train_data)))
    print("\tValidation: %d" %(len(val_data)))
    print("\tTest: %d" %(len(test_data)))

    augmentation_val = ["none"]
    if utilConst.AUGMENTATION_RANDOM in config.aug:
        augmentation_val = ["random"]
    
    #model = CNNmodel.get_model(input_shape, config.no_mask, config.n_la, config.nb_fil, config.ker, dropout=config.drop, stride=2)
    
    train_generator = util.create_generator(train_data, config.no_mask, 1, input_shape, config.n_pa, config.n_an, config.aug, config.ink_rate)
    val_generator = util.create_generator(val_data, config.no_mask, 1, input_shape, config.n_pa, config.n_an, augmentation_val, config.ink_rate)
    
    nb_train_pages = len(train_data)
    nb_val_pages = len(val_data)
    
    number_annotated_patches_train = util.get_number_annotated_patches(train_data, input_shape[0], input_shape[1], config.n_pa)  
    number_annotated_patches_val = util.get_number_annotated_patches(val_data, input_shape[0], input_shape[1], config.n_pa)  
    
    bg_full_count = 0
    ink_full_count = 0

    bg_count = 0
    ink_count = 0

    n_annotated_patches_real_count = 0
    n_annotated_patches_real_full_count = 0

    list_n_annotated_patches = []
    list_n_annotated_patches_full = []
    
    all_dataset = train_data+val_data+test_data

    list_rows = []
    list_cols = []

    import copy
    for data_item in train_data:

        gr, gt, regions_mask, n_annotated_patches_real = util.get_image_with_gt(data_item[0], data_item[1], config.n_an, input_shape[0], input_shape[1], 1, config.ink_rate, True)
        _, gt_full, regions_mask_full, n_annotated_patches_real_full = util.get_image_with_gt(data_item[0], data_item[1], 0, input_shape[0], input_shape[1], 1, config.ink_rate, True)
        
        gt_masked = copy.deepcopy(gt)
        gt_masked[np.where(regions_mask==0)] = -100

        gt_masked_full = copy.deepcopy(gt_full)
        gt_masked_full[np.where(regions_mask_full==0)] = -100


        rows, cols, channels = gr.shape
        list_rows.append(rows)
        list_cols.append(cols)

        
        bg_full = np.sum(gt_masked_full==0)
        ink_full = np.sum(gt_masked_full==1)

        bg_full_count += bg_full
        ink_full_count += ink_full

        if (bg_full+ink_full) == 0:
            rate_full = 0
        else:
            rate_full = ink_full / (bg_full+ink_full)
        

        bg = np.sum(gt_masked==0)
        ink = np.sum(gt_masked==1)
        if (bg + ink) == 0:
            rate = 0
        else:
            rate = ink / (bg+ink)

        bg_count += bg
        ink_count += ink

        n_annotated_patches_real_count += n_annotated_patches_real
        n_annotated_patches_real_full_count += n_annotated_patches_real_full
        
        print(data_item[0])
        print ("Annotated (Ink/BG): " +number_to_string(ink) +"/" +number_to_string(bg) +" - " + number_to_string(rate,5))
        print ("Full-page (Ink/BG): " +number_to_string(ink_full) +"/" +number_to_string(bg_full) +" - " + number_to_string(rate_full,5))
        print("Number of samples (Annotated / Full-page): " + number_to_string(n_annotated_patches_real) + " / " + number_to_string(n_annotated_patches_real_full))

        list_n_annotated_patches.append(n_annotated_patches_real)
        list_n_annotated_patches_full.append(n_annotated_patches_real_full)

        pass          

    if (bg_count+ink_count) == 0:
        rate_count = 0
    else:
        rate_count = ink_count / (bg_count+ink_count)
    
    if (bg_full_count+ink_full_count) == 0:
        rate_full_count = 0
    else:
        rate_full_count = ink_full_count / (bg_full_count+ink_full_count)

    min_rows = np.min(list_rows)
    max_rows = np.max(list_rows)
    avg_rows = np.average(list_rows)

    min_cols = np.min(list_cols)
    max_cols = np.max(list_cols)
    avg_cols = np.average(list_cols)

    list_cols = []

    print('*'*80)
    print("Number of pages: " + number_to_string(len(all_dataset)))
    print("Resolution (rows x cols) (MIN, MAX, AVG): " + number_to_string(min_rows)+"x"+number_to_string(min_cols) +", " + number_to_string(max_rows) + "x"+number_to_string(max_cols) + ", " + number_to_string(avg_rows,0) + "x" + number_to_string(avg_cols,0))
    print("Annotations per page: " + number_to_string(config.n_an))

    print ("Annotated (Ink/BG - RATE): " +number_to_string(ink_count) +" / " +number_to_string(bg_count) +" - " + number_to_string(rate_count,5))
    print ("Full-page (Ink/BG - RATE): " +number_to_string(ink_full_count) +"/" +number_to_string(bg_full_count) + " - " + number_to_string(rate_full_count,5))
    print("Number of samples (Annotated / Full-page): " + number_to_string(n_annotated_patches_real_count) + " / " + number_to_string(n_annotated_patches_real_full_count))

    average_annotated_patches_full = np.average(list_n_annotated_patches_full)
    min_annotated_patches_full = np.min(list_n_annotated_patches_full)
    max_annotated_patches_full = np.max(list_n_annotated_patches_full)
    sum_annotated_patches_full = np.sum(list_n_annotated_patches_full)
    print("Patches per page:")
    print ("MIN - MAX - AVG: " + number_to_string(min_annotated_patches_full) + " - " + number_to_string(max_annotated_patches_full) + " - " + number_to_string(average_annotated_patches_full,1)+ " - " + number_to_string(sum_annotated_patches_full,0))

    average_annotated_patches = np.average(list_n_annotated_patches)
    min_annotated_patches = np.min(list_n_annotated_patches)
    max_annotated_patches = np.max(list_n_annotated_patches)
    sum_annotated_patches = np.sum(list_n_annotated_patches)
    print("Patches ANNOTATED AND USED per page:")
    print ("MIN - MAX - AVG - SUM: " + number_to_string(min_annotated_patches) + " - " + number_to_string(max_annotated_patches) + " - " + number_to_string(average_annotated_patches,1) + " - " + number_to_string(sum_annotated_patches,0))


    print("Summary")
    #datasets/Dibco/train/SRC;1;0,23;15105;50431;0,23048;0;1;0,0;1;0;1;0,0;1;
    str_result = config.db_train_src + ";" 
    str_result += number_to_string(config.n_an)+ ";" 
    str_result += number_to_string(config.ink_rate, 5) + ";" 
    str_result += number_to_string(ink_count) + ";" 
    str_result += number_to_string(bg_count) + ";" 
    str_result += number_to_string(rate_count, 5) + ";"
    str_result += number_to_string(min_annotated_patches,0)+ ";" 
    str_result += number_to_string(max_annotated_patches,0)+ ";" 
    str_result += number_to_string(average_annotated_patches,0)+ ";" 
    str_result += number_to_string(sum_annotated_patches,0) + ";" 
    str_result += number_to_string(min_annotated_patches_full,0)+ ";" 
    str_result += number_to_string(max_annotated_patches_full,0)+ ";" 
    str_result += number_to_string(average_annotated_patches_full,0)+ ";" 
    str_result += number_to_string(sum_annotated_patches_full,0)
    str_result += ";"
    print(str_result)
    pass
    
    if config.res is not None:
        utilIO.appendString(str_result, config.res, True)
    
      