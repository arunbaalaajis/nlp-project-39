# Incorporating Fusion Techniques into HotPot Baseline model
The HotPotQA baseline model artchitecture uses Linear_Sum to concatenate outputs of different branches, before sending them to the next layer.
We have incorporated 3 different fusion techniques, namely:
1. Multi Modal Tucker Fusion
2. Multi Modal Factorized Bilinear Pooling (MFB)
3. Combination of Tucker and MFB

Our task is to evaluate these models on the HotPotQA dataset.


In order to run the Fusion Experiments, copy the modified run, model and main files into the baseline_code directory

```
cp Fusion_main.py Fusion_model.py Fusion_run.py ../../baseline_code
```
Note: All the preprocessing steps are the same as the baseline approach, and can be done using the original main.py file or the updated Fusion_main.py file.


## Training

To train the Fusion model, run the following command (`--fusion` can take arguments `none` , `MFB` , `Tucker` and `Multi`) from the baseline_code directory  

```
CUDA_VISIBLE_DEVICES=0 python Fusion_main.py --mode train \
--para_limit 2250 --batch_size 48 --init_lr 0.1 --keep_prob 1.0 \ 
--sp_lambda 1.0 --fusion none
```

## Local Evaluation

First, make predictions and save the predictions into a file (replace `--save` with your own file name, and `--fusion` with the Fusion method used in the particular trained model).

```
CUDA_VISIBLE_DEVICES=0 python Fusion_main.py --mode test --data_split dev --para_limit 2250 --batch_size 24 --init_lr 0.1 \ 
--keep_prob 1.0 --sp_lambda 1.0 --save HOTPOT-20180924-160521 --prediction_file dev_distractor_pred.json --fusion none
```

Then call the evaluation script:
```
python hotpot_evaluate_v1.py dev_distractor_pred.json hotpot_dev_distractor_v1.json
```
