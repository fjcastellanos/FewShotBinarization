


import sys, os, re
import shutil
import cv2
import numpy as np
from os.path import dirname
import copy


def createParentDirectory(path_file):
    pathdir = dirname(path_file)
    mkdirp(pathdir)

def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def deleteFolder(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)



def listFilesRecursive(path_dir):
    
    try:
        listOfFile = os.listdir(path_dir)
    except Exception:
        pathdir_exec = os.path.dirname(os.path.abspath(__file__))
        path_dir = pathdir_exec + "/" + path_dir
        listOfFile = os.listdir(path_dir)

    list_files = list()
    
    for entry in listOfFile:
        fullPath = os.path.join(path_dir, entry)
        if os.path.isdir(fullPath):
            list_files = list_files + listFilesRecursive(fullPath)
        else:
            list_files.append(fullPath)
    
    list_files.sort()            
    return list_files


def match_SRC_GT_Images(list_src_images, list_gt_images):
    list_matched_data = []
    for idx_image in range(len(list_src_images)):
        src_image = list_src_images[idx_image]
        gt_image = list_gt_images[idx_image]

        src_basename = os.path.basename(src_image)
        gt_basename = os.path.basename(gt_image)

        print('*'*80)
        print("Image %d:" % (idx_image))
        print("\t%s" % (src_image))
        print("\t%s" % (gt_image))

        print("src_basename = " + src_basename)
        print("gt_basename = " + gt_basename)
        assert(src_basename == gt_basename)

        list_matched_data.append( (src_image, gt_image))
        

    print("SRC and GT images are match.")
    return list_matched_data


def load_gt_image(path_file, regions_mask=None):
    file_img = cv2.imread(path_file, cv2.IMREAD_UNCHANGED,)  # 4-channel
    if file_img is None : 
        raise Exception(
            'It is not possible to load the image\n'
            "Path: " + str(path_file)
        )
    
    return file_img

def load_src_image(path_file, mode=cv2.IMREAD_COLOR):

    file_img = cv2.imread(path_file, mode)
    if file_img is None : 
        raise Exception(
            'It is not possible to load the image\n'
            "Path: " + str(path_file)
        )

    return file_img


def saveImage (image, path_file):
    assert 'numpy.ndarray' in str(type(image))
    assert type(path_file) == str
    
    path_dir = dirname(path_file)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir, 493)

    cv2.imwrite(path_file, image)


def appendString(content_string, path_file, close_file = True):
    assert type(content_string) == str
    assert type(path_file) == str
    
    path_dir = dirname(path_file)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir, 493)
            
    f = open(path_file,"a")
    f.write(content_string + "\n")
    
    if close_file == True:
        f.close()
    
        
        
def __remove_attribute_namespace(config, key):
    try:
        delattr(config, key)
    except:
        pass

def __compact_number(value):
    text = '%g' % value
    return text[1:] if text.startswith('0.') else text


def getPathModel(config):
    config_copy = copy.deepcopy(config)
    
    __remove_attribute_namespace(config_copy, 'test')
    
    __remove_attribute_namespace(config_copy, 'db_test_src')
    __remove_attribute_namespace(config_copy, 'db_test_gt')
    __remove_attribute_namespace(config_copy, 'gpu')
    __remove_attribute_namespace(config_copy, 'verbose')
    __remove_attribute_namespace(config_copy, 'res')
    __remove_attribute_namespace(config_copy, 'save')
    __remove_attribute_namespace(config_copy, 'aug_test')
    __remove_attribute_namespace(config_copy, 'n_aug')
    __remove_attribute_namespace(config_copy, 'drop_test')
    __remove_attribute_namespace(config_copy, 'all_ths')

    # Preserve byte-for-byte legacy checkpoint paths when mixing is disabled.
    if getattr(config_copy, 'sae_mix', 'none') == 'none':
        __remove_attribute_namespace(config_copy, 'sae_mix')
    if getattr(config_copy, 'sae_mix', 'none') != 'mixup':
        __remove_attribute_namespace(config_copy, 'mixup_min')
        __remove_attribute_namespace(config_copy, 'mixup_max')

    # Alternative samplers (reviewer request) never alter legacy checkpoint
    # paths: their options are removed here and encoded as a compact suffix.
    sae_sampling = getattr(config_copy, 'sae_sampling', 'default')
    sampling_suffix = ''
    if sae_sampling == 'centered_context':
        sampling_suffix = '__ctx%s-%s' % (__compact_number(config.context_min), __compact_number(config.context_max))
    elif sae_sampling == 'output_masking':
        sampling_suffix = '__rom%s-%sb%d' % (__compact_number(config.mask_min), __compact_number(config.mask_max), config.mask_block)
    for key in ('sae_sampling', 'context_min', 'context_max', 'mask_min', 'mask_max', 'mask_block'):
        __remove_attribute_namespace(config_copy, key)

    if config.ink_rate == 0.0025:
        __remove_attribute_namespace(config_copy, 'ink_rate')

    if config.no_mask is None or config.no_mask == False:
        __remove_attribute_namespace(config_copy, 'no_mask')

    
    str_config = str(config_copy).replace("Namespace", "modelCNN_").replace("(", "").replace(")", "").replace("=", "_").replace("'", "").replace(",","").replace(" ", "__").replace("[", "_").replace("]","_").replace("]","_").replace("/", "_")
    str_config = "models/modelCNN/"+str_config + ".h5"
    str_config = str_config.replace("flip","f").replace("rot", "ro").replace("scale", "sc")
    str_config = str_config.replace("datasets","dbs").replace("training", "train").replace("db_train","dtr").replace("pages_train", "pt")
    # Preserve the historical Namespace-derived checkpoint naming convention.
    # Only abbreviate the NEW mixing fields, never rename legacy fields.
    # Parameters are still present in the filename (including MixUp bounds).
    mix = getattr(config_copy, 'sae_mix', 'none')
    if mix != 'none':
        str_config = str_config.replace('sae_mix_mixup', 'mu')
        str_config = str_config.replace('sae_mix_cutmix', 'cm')
        str_config = str_config.replace('mixup_min', 'lo')
        str_config = str_config.replace('mixup_max', 'hi')
        filename = os.path.basename(str_config)
        if mix == 'mixup' and len(os.fsencode(filename)) > 255:
            # Only compact oversized MixUp names. Keep previous checkpoint
            # paths unchanged whenever they already fit on the filesystem.
            # The 'mu' marker is already present later in the name.
            def compact_bounds(match):
                high, low = match.group(1), match.group(2)
                low = low[1:] if low.startswith('0.') else low
                high = high[1:] if high.startswith('0.') else high
                return '__a%s-%s' % (low, high)

            str_config = re.sub(r'__hi_([^_]+)__lo_([^_]+)', compact_bounds, str_config, count=1)

            # Preserve already existing compact checkpoint paths. Only when
            # still oversized, remove one more redundant character from the
            # bounds marker (the 'mu' suffix identifies MixUp unambiguously).
            if len(os.fsencode(os.path.basename(str_config))) > 255:
                str_config = str_config.replace('__a.', '__.', 1)

        filename = os.path.basename(str_config)
        if len(os.fsencode(filename)) > 255:
            raise ValueError('Checkpoint filename still exceeds 255 bytes (%s): %s' % (len(os.fsencode(filename)), filename))
    if sampling_suffix:
        str_config = str_config[:-len('.h5')] + sampling_suffix + '.h5'
        filename = os.path.basename(str_config)
        if len(os.fsencode(filename)) > 255:
            raise ValueError('Checkpoint filename exceeds 255 bytes (%s): %s' % (len(os.fsencode(filename)), filename))
    return str_config
