#!/bin/bash

MODE="test"
GPU=0
TYPE="CNN"

DB_TRAIN_SRC=""
DB_TRAIN_GT=""

DB_TEST_SRC=""
DB_TEST_GT=""

AUG="random flipH flipV rot scale" #"random"  # 'all', 'none', 'flipH', 'flipV', 'wb', 'expos', 'rot', 'scale', 'blur', 'dropout'

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
PATHRESULTS="results/test_models_PALM_DISTORTED_CONTRAST_0.02_inkrate_Allpages_mask_random.txt"
OPTIONS="--test"    #--test



AUG_serial=${AUG// /.}
AUG_serial=${AUG_serial////-}

options_serial=${OPTIONS// /.}
options_serial=${options_serial////-}

#"sal" "dibco2016" "dibco2014" "palm0" "palm1" "phi" "ein"

RUN=1
COUNTER=0

LIST_DISTORTED_GAMMA=(
  "Palm_dist/gamma/g0.7"
  "Palm_dist/gamma/g0.8"
  "Palm_dist/gamma/g1.0"
  "Palm_dist/gamma/g1.2"
  "Palm_dist/gamma/g1.4"
)

LIST_DISTORTED_CONTRAST=(
  "Palm_dist/contrast/a0.6"
  "Palm_dist/contrast/a0.8"
  "Palm_dist/contrast/a1.0"
  "Palm_dist/contrast/a1.2"
  "Palm_dist/contrast/a1.4"
)

LIST_DISTORTED_BRIGHTNESS=(
  "Palm_dist/brightness/b-20"
  "Palm_dist/brightness/b-40"
  "Palm_dist/brightness/b+0"
  "Palm_dist/brightness/b+20"
  "Palm_dist/brightness/b+40"
)


LIST_DISTORTED_TARGET=("${LIST_DISTORTED_CONTRAST[@]}")

mkdir logs/${MODE}/
for FILTERS in 32; do
    for WINDOW_W in 256; do
        WINDOW_H=${WINDOW_W}
        for LAYERS in 4; do
            for KERNEL_SIZE in 3; do
                for DROPOUT in 0.2; do
                    for PAGES_TRAIN in -1; do
                        for NUMBER_ANNOTATED_PATCHES in 32; do
                            for NUMBER_PATCHES in 1024; do #1 2 4 8 16 32 64 128 256 512 1024
                                for source in "Palm" ; do #"Dibco" "Einsiedeln" "Palm" "PHI" "Salzinnes"
	                                for target in "${LIST_DISTORTED_TARGET[@]}"; do
						TARGET_serial=${target// /.}
						TARGET_serial=${TARGET_serial////-}
			                    for ink_th in 0.02; do

			                        output_file="logs/${MODE}/out_${TYPE}_inkth_${ink_th}_S-${source}_T-${TARGET_serial}_aug${AUG_serial}_w${WINDOW_W}_h${WINDOW_H}_l${LAYERS}_f${FILTERS}_k${KERNEL_SIZE}_d${DROPOUT}_pt${PAGES_TRAIN}_np${NUMBER_PATCHES}_nap${NUMBER_ANNOTATED_PATCHES}_e${EPOCHS}_b${BATCH_SIZE}_${options_serial}.txt"
			                        echo $output_file

			                        db_path_train_src="datasets/"${source}"/train/SRC"
			                        db_path_train_gt="datasets/"${source}"/train/GT"
			                        
			                        db_path_test_src="datasets/"${target}"/test/SRC"
			                        db_path_test_gt="datasets/"${target}"/test/GT"

			                        let COUNTER++

			                        echo ${COUNTER}"\n"${RUN}

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
												--all_ths \
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
done


