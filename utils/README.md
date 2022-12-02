# Fine tuning BERT model using huggingface trainers
Our model makes use of bert-base-uncased model to generate encoded representations for a given input sentence.
Our task is to achieve multi-hop reasoning on HotPotQA dataset.
HotPotQA dataset is built on top of wikipedia paragraphs.
Hence we plan on fine tuning our BERT model on WikiText2 corpus https://huggingface.co/datasets/wikitext
In order to finetune run:
```
python run_clm.py \
    --model_name_or_path bert-base-uncased \
    --train_file wiki_text_2/ \
    --do_train \
    --do_eval \
    --output_dir  out_dir
```

To use the fine-tuned model: Please download the model from our google drive link: https://drive.google.com/drive/folders/1AnX0Xj-MW3FMcX2A12vS_2qYinVipVct?usp=sharing

Reference:
1. Huggingface run_clm trainer: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

