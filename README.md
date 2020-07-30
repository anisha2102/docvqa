# Document Visual Question Answering (DocVQA)
This repo hosts the basic functional code for our approach entitled [HyperDQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&view=method_info&task=1&m=75548) in the [Document Visual Question Answering](https://rrc.cvc.uab.es/?ch=17) competition hosted as a part of [Workshop on Text and Documents in Deep Learning Era](https://cvpr2020text.wordpress.com) at [CVPR2020](http://cvpr2020.thecvf.com). Our approach stands at position 4 on the [Leaderboard](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1).

Read more about our approach in this [blogpost](https://medium.com/@anishagunjal7/document-visual-question-answering-e6090f3bddee)!

## Installation
### Virtual Environment Python 3 (Recommended)
1) Clone the repository
```
git clone https://github.com/anisha2102/docvqa.git
```

2) Install libraries
```
pip install -r requirements.txt
```

## Downloads
1) Download the dataset
The dataset for Task 1 can be downloaded from the Competition [Website](https://rrc.cvc.uab.es/?ch=17) from the Downloads Section.
The dataset consists of document images and their corresponding OCR transcriptions.

2) Download the pretrained model
Download the pretrained model for LayoutLM-Base, Uncased from [here](https://github.com/microsoft/unilm/tree/master/layoutlm)
## Prepare dataset
```
python create_dataset.py \
         <data-ocr-folder> \
         <data-documents-folder> \
         <path-to-train_v1.0.json> \
         <train-output-json-path> \
         <validation-output-json-path>
```
## Train the model
```
CUDA_VISIBLE_DEVICES=0 python run_docvqa.py \
    --data_dir <data-folder> \
    --model_type layoutlm \
    --model_name_or_path <pretrained-model-path> \ #example ./models/layoutlm-base-uncased
    --do_lower_case \
    --max_seq_length 512 \
    --do_train \
    --num_train_epochs 15 \
    --logging_steps 500 \
    --evaluate_during_training \
    --save_steps 500 \
    --do_eval \
    --output_dir  <data-folder>/<exp-folder> \
    --per_gpu_train_batch_size 8 \
    --overwrite_output_dir \
    --cache_dir <data-folder>/models \
    --skip_match_answers \
    --val_json <train-output-json-path> \
    --train_json <train-output-json-path> \
```
## Model Checkpoints
Download the pytorch_model.bin file from the link below and copy it to the models folder.
[Google Drive Link](https://drive.google.com/file/d/1W4E06nb-tDcjKVN9iCjjk0b_3EyHkqVr/view?usp=sharing)

## Demo
Try out the demo on a sample datapoint with demo.ipynb

## Acknowledgements
The code and pretrained models are based on [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) and [HuggingFace Transformers](https://github.com/huggingface/transformers). Many thanks for their amazing open source contributions.
