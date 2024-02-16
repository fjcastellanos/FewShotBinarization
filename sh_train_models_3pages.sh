#!/bin/bash

MODE="train"
GPU=0
TYPE="CNN"

DB_TRAIN_SRC=""
DB_TRAIN_GT=""

DB_TEST_SRC=""
DB_TEST_GT=""

AUG="flipH flipV rot scale" #"random"  # 'all', 'none', 'flipH', 'flipV', 'wb', 'expos', 'rot', 'scale', 'blur', 'dropout'


WINDOW_W=256
WINDOW_H=256

LAYERS=4
FILTERS=64
KERNEL_SIZE=5
DROPOUT=0.2

PAGES_TRAIN=-1
NUMBER_PATCHES=2048
NUMBER_ANNOTATED_PATCHES=1

EPOCHS=200
BATCH_SIZE=32
VERBOSE=1
PATHRESULTS="results/tests_sh_train_models_1.txt"
OPTIONS="-no_mask"    #--test



AUG_serial=${AUG// /.}
AUG_serial=${AUG_serial////-}

options_serial=${OPTIONS// /.}
options_serial=${options_serial////-}

#"sal" "dibco2016" "dibco2014" "palm0" "palm1" "phi" "ein"

RUN=1
COUNTER=0

mkdir logs/${MODE}/
for FILTERS in 32; do
    for WINDOW_W in 256; do
        WINDOW_H=${WINDOW_W}
        for LAYERS in 4; do
            for KERNEL_SIZE in 3; do
                for DROPOUT in 0.2; do
                    for PAGES_TRAIN in 3; do
                        for NUMBER_ANNOTATED_PATCHES in -1 32 16 8 4 2 1 ; do
                            for NUMBER_PATCHES in 0; do #1 2 4 8 16 32 64 128 256 512 1024
								NUMBER_PATCHES=$NUMBER_ANNOTATED_PATCHES
                                for source in "Dibco" "Einsiedeln" "Palm" "PHI" "Salzinnes" ; do #"Dibco" "Einsiedeln" "Palm" "PHI" "Salzinnes"
                                    for ink_th in 0.02; do 
                                        target=${source}

                                        output_file="logs/${MODE}/out_${TYPE}_inkth_${ink_th}_${source}_aug${AUG_serial}_w${WINDOW_W}_h${WINDOW_H}_l${LAYERS}_f${FILTERS}_k${KERNEL_SIZE}_d${DROPOUT}_pt${PAGES_TRAIN}_np${NUMBER_PATCHES}_nap${NUMBER_ANNOTATED_PATCHES}_e${EPOCHS}_b${BATCH_SIZE}_${options_serial}.txt"
                                        echo $output_file

                                        db_path_train_src="datasets/"${source}"/train/SRC"
                                        db_path_train_gt="datasets/"${source}"/train/GT"
                                        
                                        db_path_test_src="datasets/"${target}"/test/SRC"
                                        db_path_test_gt="datasets/"${target}"/test/GT"

                                        let COUNTER++

                                        echo ${COUNTER}"\n"${RUN}

                                        if [[ $NUMBER_PATCHES -eq 512 && $PAGES_TRAIN -eq 1 && $NUMBER_ANNOTATED_PATCHES -eq -1 &&  "$source" = "Einsiedeln" && $ink_th -eq 0.0 ]] ; then
                                            let RUN=1
                                        fi
                                        if [ ${RUN} -eq 1 ] ; then
                                            echo 'RUN!!!\n'
                                            python -u main.py \
                                                    -db_train_src ${db_path_train_src} \
                                                    -db_train_gt ${db_path_train_gt} \
                                                    -db_test_src ${db_path_test_src} \
                                                    -db_test_gt ${db_path_test_gt} \
                                                    -aug ${AUG} \
                                                    -window_w ${WINDOW_W} \
                                                    -window_h ${WINDOW_H} \
                                                    -l ${LAYERS} \
                                                    -f ${FILTERS} \
                                                    -k ${KERNEL_SIZE} \
                                                    -drop ${DROPOUT} \
                                                    -pages_train ${PAGES_TRAIN} \
                                                    -npatches   ${NUMBER_PATCHES} \
                                                    -n_annotated_patches ${NUMBER_ANNOTATED_PATCHES} \
                                                    -e ${EPOCHS} \
                                                    -b ${BATCH_SIZE} \
                                                    -verbose ${VERBOSE} \
                                                    -gpu ${GPU} \
                                                    -res ${PATHRESULTS} \
                                                    -ink_rate ${ink_th} \
                                                    ${OPTIONS} \
                                                    &> ${output_file}
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done        
        done
    done
done


