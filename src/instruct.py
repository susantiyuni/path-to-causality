import os, sys, json, logging, argparse, time
root_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.insert(0, root_path)

from tqdm import tqdm
import llm as llm
# import icl_templatizer as icl
import zero_templatizer as zero
import utils as u

from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=203)
  parser.add_argument("--task", type=str, default="")
  # parser.add_argument("--input_path", type=str, default="")
  parser.add_argument("--train_path", type=str, default="")
  parser.add_argument("--test_path", type=str, default="")
  parser.add_argument("--output_path", type=str, default="checkpoints/")
  parser.add_argument("--kg_path", type=str, default="")
  parser.add_argument("--node_mapping", type=str, default="")
  parser.add_argument("--model_name", type=str, default="mistral")
  parser.add_argument("--template_id", type=str, default="tmc")
  parser.add_argument('--max_new_tokens', type=int, default=5)
  parser.add_argument('--text_label', type=str, default="false,true")
  parser.add_argument('--k_sample', type=int, default=4) ##instruct_type=fewshot
  parser.add_argument('--n_subgraph', type=int, default=4) ##instruct_type=fewshot
  parser.add_argument('--subgraph_mode', type=str, default='topk') ##instruct_type=fewshot
  parser.add_argument('--is_shuffle', action='store_false') ##instruct_type=fewshot
  parser.add_argument('--save_result', action='store_true')
  return parser

def main():
  args = _get_parser().parse_args()
  u.set_seed(args.seed)
  if args.save_result: 
    u.folder_check(args.output_path)
    log_file = args.output_path+"/log.log"
    with open(args.output_path+'/params.txt', 'w') as f:
      json.dump(args.__dict__, f, indent=2)
  else: log_file = "log.log"
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

  start_time = time.time()
  print ('\n')
  logging.info (f'Start: {start_time=}')
  logging.info (f'{json.dumps(vars(args), indent=2, sort_keys=False)}\n')
  instruct_type = 'zeroshot'
  if args.k_sample > 0:
    instruct_type = 'fewshot'
  if instruct_type == 'fewshot':
    data_files = {"train": args.train_path, "test": args.test_path}
    # data_files = {"train": args.input_path+"/train_full.jsonl", "test": f"{args.input_path}/{args.test_file}"}
  else: 
    # data_files = {"test": f"{args.input_path}/{args.test_file}"}
    data_files = {"test": args.test_path}
  dataset = load_dataset('json', data_files=data_files)
  classes = args.text_label.split(',')
  dataset = dataset.map( lambda x: {"text_label": [classes[int(label)] for label in x["ground_truth"]]}, batched=True, num_proc=1)
  model, tokenizer = llm.load_models(args.model_name)
  if args.node_mapping != '':
    node_mapping = u.load_jsonl(args.node_mapping)

  if instruct_type == 'fewshot':
    if args.is_shuffle:
      dataset['train'].shuffle(seed=args.seed)
      logging.info (f'{args.is_shuffle=}')
    train_k_sample = u.get_train_k_sample(dataset['train'], args.k_sample, is_shuffle=args.is_shuffle)
    # prefixed_train_prompt = icl.generate_prompt_train(train_k_sample, args.n_subgraph, args.kg_path, node_mapping, args.template_id)
    logging.info (f'## #### START TRAIN PROMPT #### ##')
    logging.info (prefixed_train_prompt)
    logging.info (f'## #### END TRAIN PROMPT #### ##')
  else:
    prefixed_train_prompt = ''
    logging.info (f'No train sample: {instruct_type}')

  trues, preds, preds_text  = [], [], []
  for i, row in enumerate(tqdm(dataset['test'], desc=f'## Processing')):
    # if i <= 2:
    # try:
    logging.info (f'Input {i=}: {row}')
    if instruct_type == 'fewshot':
      # test_prompt = icl.generate_prompt_test(row, args.n_subgraph, args.kg_path, node_mapping, args.template_id)
    elif instruct_type == 'zeroshot':
      test_prompt = zero.generate_prompt(row, args.task, args.n_subgraph, args.template_id, args.subgraph_mode, args.model_name)
    else: test_prompt = input_prompt
    logging.info (f'## #### START TEST PROMPT #### ##')
    logging.info (test_prompt)
    logging.info (f'## #### END TEST PROMPT #### ##')
    final_prompt = prefixed_train_prompt+test_prompt
    # logging.info (f'Inference..')
    llm_output = llm.inference(model, tokenizer, final_prompt, max_new_tokens=args.max_new_tokens)
    pred = u.verbalizer(llm_output.lower())
    preds.append(pred)
    llm_output = ''.join(llm_output.split())
    preds_text.append(llm_output)
    trues.append(int(row['ground_truth']))
    logging.info (f"Result {i=}: {row['text_label']=} {llm_output=} {pred=}")
    logging.info ('############################################')
    # except:
    #   logging.info (f'ERROR: {i=}')
    #   continue

  binary_all_metric = precision_recall_fscore_support(trues, preds, average='binary')
  micro_all_metric = precision_recall_fscore_support(trues, preds, average='micro')
  logging.info (f'{binary_all_metric=} {micro_all_metric=}')

  if args.save_result: u.save_result(args.output_path, trues, preds, preds_text, binary_all_metric, micro_all_metric)

  logging.info ("--- Total time: %.2f seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
  main()

  # ds = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")
  # with open('datasets/ade_all.jsonl', 'w') as fw:
  #   for i, x in enumerate(ds['train']):
  #     dic = {}
  #     dic['sentence'] = x['text']
  #     dic['relation'] = "1"
  #     dic['e1'] = x['drug']
  #     dic['e2'] = x['effect']
  #     jout = json.dumps(dic) + '\n'
  #     fw.write(jout)