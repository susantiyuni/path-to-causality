import json, sys, logging
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AdamW
import torch
from tqdm import tqdm
from rank_loss import RankLoss
import numpy as np
import os
import argparse
import dt_utils as du
import ltr_eval as leval
import utils as u

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='roberta-base')
  parser.add_argument('--loss_type', type=str, default='rank_net')
  parser.add_argument('--train_path', type=str, default='')
  parser.add_argument('--test_path', type=str, default='')
  parser.add_argument('--checkpoint_path', type=str, default='')
  parser.add_argument('--do_train', type=bool, default=False)
  parser.add_argument('--epoch', type=int, default=10)
  parser.add_argument('--neg_num', type=int, default=5) #4 for 5 paths
  parser.add_argument('--include_node_labels', type=bool, default=False)
  parser.add_argument('--include_rel_types', type=bool, default=False)
  parser.add_argument('--do_eval', type=bool, default=False)

  args = parser.parse_args()

  return args

def train(args):
  accelerator = Accelerator(gradient_accumulation_steps=8)
  neg_num = args.neg_num

  # Create cross encoder model
  config = AutoConfig.from_pretrained(args.model_name)
  config.num_labels = 1
  model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
  tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

  # Load data and permutation
  data = [json.loads(line) for line in open(args.train_path)]
  # response = json.load(open(permutation))
  # data = receive_response(data, response) #ordered according to gpt 
  dataset = du.RankingData(data, tokenizer, neg_num=neg_num, include_node_labels=args.include_node_labels, include_rel_types=args.include_rel_types)
  print (dataset[0])
  # (['Bax T47D', 'Bax T47D', 'Bax T47D', 'Bax T47D', 'Bax T47D'], ['BAX - PMAIP1 - Mitoxantrone - Breast cancer stage IV', 'BAX - breast cancer - Mitoxantrone - Breast cancer stage IV', 'BAX - GABARAP - Fulvestrant - Breast cancer stage IV', 'BAX - HSPA8 - Fulvestrant - Breast cancer stage IV', 'BAX - THAP11 - Fulvestrant - Breast cancer stage IV'], [4.08797, 4.19727, 4.52694, 4.60936, 4.72188])

  data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn,batch_size=1, shuffle=True, num_workers=0)
  optimizer = AdamW(model.parameters(), 5e-5)
  model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
  loss_function = getattr(RankLoss, args.loss_type)

  best_acc = 0.0
  ep_acc = 0.0
  best_all = {}
  best_run = 0
  for epoch in range(args.epoch):
    accelerator.print(f'Training epoch: {epoch}')
    accelerator.wait_for_everyone()
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))
    loss_report = []
    for batch in tk0:
      with accelerator.accumulate(model):
        # out = model(**batch)
        out = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = out.logits
        # logits = logits.view(-1, neg_num + 1)
        logits = logits.view(-1, neg_num)

        # y_true = torch.tensor([[1 / (i + 1) for i in range(logits.size(1))]] * logits.size(0)).cuda() #rank
        # y_true, indices = torch.sort(batch['rel_score'], descending=True)
        y_true = batch['rel_score']
        loss = loss_function(logits, y_true)

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()
        loss_report.append(accelerator.gather(loss).mean().item())
      tk0.set_postfix(loss=sum(loss_report) / len(loss_report))
    accelerator.wait_for_everyone()
  
    all_metrics = eval(args, epoch, model, tokenizer)
    logging.info (f"{epoch=} {all_metrics=} \n")
    ep_acc = all_metrics['NDCG@5']
    if ep_acc > best_acc:
      best_acc = ep_acc
      best_all = all_metrics
      best_run = epoch
      logging.info (f"Congratulations! New best accuracy: {best_acc}")
      # Save model
      unwrap_model = accelerator.unwrap_model(model)
      # os.makedirs(args.checkpoint_path, exist_ok=True)
      unwrap_model.save_pretrained(args.checkpoint_path)
  
  logging.info (f"# BEST: {best_acc} # All: {best_all} # Epoch: {best_run}\n")

  return model, tokenizer, best_all

def eval(args, epc, model=None, tokenizer=None):
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'

  if model is None or tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model = model.cuda()
  model.eval()

  data = [json.loads(line) for line in open(args.test_path)]
  du.write_file(data, args.checkpoint_path+'/truth.eval', is_truth=True)

  pred_ranked = f"{args.checkpoint_path}/test_pred_ep{str(epc)}.jsonl"
  with open(pred_ranked, 'w', encoding='utf-8') as out_file:
    reranked_data = []
    for item in tqdm(data):
      q = f"{item['e1']} {item['e2']}"
      passages = [psg['stops'] for i, psg in enumerate(item['metapaths'])][:5]
      if len(passages) == 0:
        reranked_data.append(item)
        continue
      features = tokenizer([q] * len(passages), passages, padding=True, truncation=True, return_tensors="pt", max_length=500)
      features = {k: v.cuda() for k, v in features.items()}
      with torch.no_grad():
        # scores = model(**features).logits
        scores = model(features['input_ids'], attention_mask=features['attention_mask']).logits
        normalized_scores = [float(score[0]) for score in scores]
      ranked = np.argsort(normalized_scores)[::-1]
      response = ' > '.join([str(ss + 1) for ss in ranked])
      # print (ranked, response)
      ranked_result = du.receive_permutation(item, response, rank_start=0, rank_end=5)
      reranked_data.append(ranked_result)
      # print (ranked_result)
      jout = json.dumps(ranked_result) + '\n'
      out_file.write(jout)    
    # pred_out = f"{args.checkpoint_path}/pred_ep{str(epc)}.eval"
    pred_out = f"{args.checkpoint_path}/pred.eval"
    du.write_file(reranked_data, pred_out, is_truth=False)
  all_metrics = leval.run_eval(args.checkpoint_path+'/truth.eval', pred_out)
  return all_metrics
  
    
if __name__ == '__main__':
  args = parse_args()
  u.folder_check(args.checkpoint_path)
  log_file = args.checkpoint_path+"/log.log"
  with open(args.checkpoint_path+'/params.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
  for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
      logging.FileHandler(log_file, 'w+'),
      logging.StreamHandler() ] )
  logger = logging.getLogger(__name__)

  logging.info ('====Input Arguments====')
  logging.info (json.dumps(vars(args), indent=2, sort_keys=False))

  model, tokenizer = None, None
  
  if args.do_train:
    model, tokenizer, all_metrics = train(args)
  # if args.do_eval:
  #   eval(args, model, tokenizer)
