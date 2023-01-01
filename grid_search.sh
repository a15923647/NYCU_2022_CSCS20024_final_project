#!/bin/bash
set -x
for lr in "2e-7" "2e-6" "2e-5" "2e-4" "2e-3" "2e-2"; do
  for bs in 32 64 128 256 512 ; do
    sed -i "s/lr = .*/lr = ${lr}/g" config.py
    sed -i "s/BATCH_SIZE = .*/BATCH_SIZE = ${bs}/g" config.py
    python3 109550043_Final_train.py
    kaggle competitions submit -c tabular-playground-series-aug-2022 -f submission.csv -m "first submit"
    for f in $(ls v1.0_improved*train*.model); do
      python3 109550043_Final_inference.py ${f}
      kaggle competitions submit -c tabular-playground-series-aug-2022 -f submission.csv -m "${f} ${lr} ${bs} DNN model"
    done
    dirname="DNN_model_lr_${lr}_bs_${bs}_times_${t}"
    mkdir $dirname
    cp *.py $dirname
    mv *.model *.tar $dirname
    date > "${dirname}/finish_time"
  done
done

set +x
