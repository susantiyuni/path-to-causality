#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

tag_checkpoint='k0-n1-mistral-tcma-xgboost-1111'

model_name="mistralai/Mistral-7B-Instruct-v0.1"
# model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# model_name="google/gemma-7b"

test_file='test_sim.jsonl' #sentence transforer
# test_file='roberta-rnet-nl/test_pred_ep2.jsonl' #roberta-nl
# test_file='roberta-lnet-nl/test_pred_ep3.jsonl' #roberta-nl
# test_file="roberta-rmse-nl/test_pred_ep5.jsonl" #roberta-nl-rmse
# test_file="xgboost/test_pred_xgboost.jsonl" #xgboost
# test_file='test_gpt_a.jsonl' #gpt reranker


subgraph_mode='topk' #topk, random, lastk
template_id='tcma'
n_subgraph=1

# template_id='tnb'
# n_subgraph=0

# for task in "ade" "comagc" "gene" "semeval"; do
for task in "comagc"; do
  train_path="datasets/${task}/train_full.jsonl"
  test_path="checkpoints/${task}/${test_file}"
  # test_path="datasets/${task}/${test_file}" #for baseline (random, sim)
  output_path="results/${task}/${tag_checkpoint}"
  python instruct.py \
  --task ${task} \
  --train_path ${train_path} \
  --test_path ${test_path} \
  --output_path ${output_path} \
  --model_name ${model_name} \
  --template_id ${template_id} \
  --max_new_tokens 6 \
  --text_label 'non-causal,causal' \
  --k_sample 0 \
  --n_subgraph ${n_subgraph} \
  --subgraph_mode ${subgraph_mode} \
  --seed 1111 \
  --save_result
done
