#!/bin/bash

MODE="info"
GPU=0
TYPE="CNN"

DB_TRAIN_SRC="dbs/b-59-850/training/images"
DB_TRAIN_GT="dbs/b-59-850/training/layers/text"

DB_TEST_SRC="dbs/b-59-850/test/images"
DB_TEST_GT="dbs/b-59-850/test/layers/text"

AUG="random"  # 'all', 'none', 'flipH', 'flipV', 'wb', 'expos', 'rot', 'scale', 'blur', 'dropout'

EXTRACTION_MODE="man" # "man", "ent"
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
PATHRESULTS="results/info.txt"
OPTIONS=""    #--test





options_serial=${options// /.}
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
                for DROPOUT in 0.4; do
                    for PAGES_TRAIN in 1; do
                        for NUMBER_ANNOTATED_PATCHES in 1 2 4 8 16 32 64; do
                            for NUMBER_PATCHES in 1; do
                                for source in "Dibco" "Einsiedeln" "Palm" "PHI" "Salzinnes" ; do #"Dibco" "Einsiedeln" "Palm" "PHI" "Salzinnes
                                    for ink_th in 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40; do
                                        target=${source}

                                        output_file="logs/${MODE}/out_${TYPE}_inkth_${ink_th}_ext_${EXTRACTION_MODE}_${source}_aug${AUG}_w${WINDOW_W}_h${WINDOW_H}_l${LAYERS}_f${FILTERS}_k${KERNEL_SIZE}_d${DROPOUT}_pt${PAGES_TRAIN}_np${NUMBER_PATCHES}_nap${NUMBER_ANNOTATED_PATCHES}_e${EPOCHS}_b${BATCH_SIZE}_${options_serial}.txt"
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
                                            python -u info.py \
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


