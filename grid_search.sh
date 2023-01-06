#!/bin/bash
set -x
for t in $(seq 1 1); do
for lr in "5e-4" "5e-3" "5e-2" "5e-1" ; do
  for bs in 64 128 256 512 ; do
    sed -i "s/lr = .*/lr = ${lr}/g" config.py
    sed -i "s/BATCH_SIZE = .*/BATCH_SIZE = ${bs}/g" config.py
    python3 109550043_Final_train.py
    kaggle competitions submit -c tabular-playground-series-aug-2022 -f submission.csv -m "first submit"
    dirname="DNN_model_lr_${lr}_bs_${bs}_times_${t}_dc_3e-4_no_bn"
    for f in $(ls v1.0_improved*train*.model); do
      python3 109550043_Final_inference.py ${f}
      kaggle competitions submit -c tabular-playground-series-aug-2022 -f submission.csv -m "$(pwd)/${dirname}/${f} ${lr} ${bs} ${t} DNN model v3.2 grid searching"
    done
    mkdir $dirname
    cp *.py $dirname
    mv *.model *.tar $dirname
    date > "${dirname}/finish_time"
  done
done
done
set +x
