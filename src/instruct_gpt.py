import os, sys, logging, argparse, time
root_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.insert(0, root_path)
import jsonlines, json

from tqdm import tqdm
import llm as llm
import zero_templatizer as zero
import utils as u
import llm_gpt as gpt

from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=203)
  parser.add_argument("--test_path", type=str, default="")
  # parser.add_argument("--test_file", type=str, default="/test.tsv")
  parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
  parser.add_argument("--kg_path", type=str, default="")
  parser.add_argument("--node_mapping", type=str, default="")
  parser.add_argument("--model_name", type=str, default="gpt-4")
  parser.add_argument("--template_id", type=str, default="tmc")
  parser.add_argument('--max_new_tokens', type=int, default=5)
  parser.add_argument('--text_label', type=str, default="non-causal,causal")
  parser.add_argument('--k_sample', type=int, default=4) ##instruct_type=fewshot
  parser.add_argument('--n_subgraph', type=int, default=4) ##instruct_type=fewshot
  parser.add_argument('--is_shuffle', action='store_false') ##instruct_type=fewshot
  parser.add_argument('--save_result', action='store_true')
  return parser

def main():
  args = _get_parser().parse_args()
  u.set_seed(args.seed)
  if args.save_result: 
    u.folder_check(args.checkpoint_path)
    log_file = args.checkpoint_path+"/log.log"
    with open(args.checkpoint_path+'/params.txt', 'w') as f:
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
  logging.info (f'{args=}\n')
  # data_files = {"test": args.input_path+args.test_file}
  # data_files = {"test": args.input_path+"/test_evs_2.tsv"}
   data_files = {"test": args.test_path}
  dataset = load_dataset("json", data_files=data_files, delimiter='\t')
  classes = args.text_label.split(',')
  dataset = dataset.map( lambda x: {"text_label": [classes[int(label)] for label in x["label"]]}, batched=True, num_proc=1)
  node_mapping = u.load_jsonl(args.node_mapping)

  trues, preds, preds_text  = [], [], []
  for i, row in enumerate(tqdm(dataset['test'], desc=f'## Processing')):
    # if i <= 1:
    try:
      logging.info ('############################################')
      logging.info (f'Input {i=}: {row}')
      test_prompt = gpt.generate_prompt(row, args.n_subgraph, args.kg_path, node_mapping, args.template_id)
      test_prompt = gpt.write_prompt(test_prompt)
      logging.info (f'## #### START TEST PROMPT #### ##')
      logging.info (test_prompt)
      logging.info (f'## #### END TEST PROMPT #### ##')
      llm_output = gpt.send_prompt(test_prompt, args.model_name)
      pred = u.verbalizer(llm_output.lower())
      preds.append(pred)
      llm_output = ' '.join(llm_output.split())
      preds_text.append(llm_output)
      trues.append(row['label'])
      logging.info (f"Result {i=}: {row['text_label']=} {llm_output=} {pred=}")
      logging.info ('############################################')
    except:
      logging.info (f'ERROR: {i=}')
      continue

  binary_all_metric = precision_recall_fscore_support(trues, preds, average='binary')
  micro_all_metric = precision_recall_fscore_support(trues, preds, average='micro')
  logging.info (f'{binary_all_metric=} {micro_all_metric=}')

  if args.save_result: u.save_result(args.checkpoint_path, trues, preds, preds_text, binary_all_metric, micro_all_metric)

  logging.info ("--- Total time: %.2f seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
  main()