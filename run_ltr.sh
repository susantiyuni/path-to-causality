export CUDA_VISIBLE_DEVICES=1

tag_checkpoint='roberta-lnet'
model_name="roberta-base" 
loss="list_net" #rank_net, pointwise_rmse

for task in "ade" "comagc" "gene" "semeval"; do
  train_path="datasets/${task}/train_full.jsonl"
  test_path="datasets/${task}/test_truth.jsonl"
  checkpoint_path="checkpoints/${task}/${tag_checkpoint}"
  python src/ltr_nn.py \
    --train_path ${train_path} \
    --test_path ${test_path} \
    --checkpoint_path ${checkpoint_path} \
    --loss_type ${loss} \
    --model_name ${model_name} \
    --do_train true \
    --epoch 10 \
    # --include_node_labels true \
done
