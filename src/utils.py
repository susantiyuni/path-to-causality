import numpy as np
import random
import jsonlines, os, itertools

def write_path(p, nl, reltypes ):
  c = [k for k in reltypes.split('_')[1] if k.islower()][0]
  rel = reltypes.split('_')[1].split(c)
  rel0, rel1 = rel[0], rel[1]
  if len(rel0) > 1: rel0=rel0[0]
  if len(rel1) > 1: rel1=rel1[0]
  newrel = rel_preposition(reltypes.split('_')[0])
  if nl[0].startswith(rel0) and nl[1].startswith(rel1):
    sent = f"{p[0]} {newrel} {p[1]}"
    # sent = p[0]+' '+newrel+' '+p[1]
    # sent = nl[0]+' '+p[0]+' '+newrel+' '+nl[1]+' '+p[1]
  else:
    sent = f"{p[1]} {newrel} {p[0]}"
    # sent = p[1]+' '+newrel+' '+p[0]
    # sent = nl[1]+' '+p[1]+' '+newrel+' '+nl[0]+' '+p[0]
  return sent

def pairwise(iterable):
  "s -> (s0, s1), (s1, s2), (s2, s3), ..."
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)  

def rel_preposition(rel):
  if rel in ['ASSOCIATES', 'COVARIES', 'INTERACTS']:
    rel = rel+' with'
  elif rel in ['LOCALIZES', 'PARTICIPATES', 'PRESENTS']:
    rel = rel+' in'
  return rel.lower()

def get_train_k_sample(train_data, k_sample, is_shuffle=True):
  pos, neg = [], []
  for row in train_data:
    if row['label'] == 1:
      pos.append(row)
    else: neg.append(row)
  train_k_sample = pos[:k_sample]+neg[:k_sample]
  if is_shuffle: np.random.shuffle(train_k_sample)
  return train_k_sample

def load_jsonl(inp):
  dic={}
  with jsonlines.open(inp, 'r') as fop:
    for line in fop:
      key = tuple(line['orig_tuple'])
      dic[key] = line['node_match']
  return dic

def fix_node_pair(a, b, node_mapping):
  tup=tuple([a,b])
  node_match=node_mapping[tup]
  if a.lower()!=node_match[0].lower() and b.lower()!=node_match[1].lower():
    a=a+' or '+node_match[0]
    b=b+' or '+node_match[1]
  elif a.lower()==node_match[0].lower() and b.lower()!=node_match[1].lower():
    a=a
    b=b+' or '+node_match[1]
  elif a.lower()!=node_match[0].lower() and b.lower()==node_match[1].lower():
    a=a+' or '+node_match[0]
    b=b
  return a, b

def verbalizer(llm_output):
  llm_output = llm_output.lower()
  neg = ['non-cau', 'false', 'negative', 'no']
  pos = ['causal', 'true', 'positive', 'yes']
  if any(v in llm_output for v in neg):
    return 0
  elif any(v in llm_output for v in pos):
    return 1
  else: return 0
  
# print (verbalizer('adafa'))

def folder_check(mpath):
  if os.path.isdir(mpath): print (f'Path: {mpath}')
  else: os.makedirs(mpath, exist_ok=True)

def save_result(output_path, y_true, y_pred, pred_label, all_bi, all_mi):
  folder_check(output_path)
  with open(output_path+'/test_scores.txt', 'w') as out:
    out.write('p,r,f binary:'+str(all_bi)+'\n')
    out.write('p,r,f micro:'+str(all_mi)+'\n')
    out.write ('TRUE\tPRED\tORIG_PRED\n')
    for true, pred, text_pred in zip(y_true, y_pred, pred_label):
      out.write (str(true)+'\t'+str(pred)+'\t'+text_pred+'\n')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def set_deterministic():
    torch.cuda.empty_cache()
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
