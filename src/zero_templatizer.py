import jsonlines, json, os
import subgraph as sub
import random, ast

remove = ["<e1>","</e1>","<e2>","</e2>"]
prefix_comagc = "Given the following information, classify the relation between the gene and disease pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n"
prefix_ade = "Given the following information, classify the relation between the drug and side effect pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n"
# prefix_gene = "Given the following information, classify the relation between the pair. If there is a cause-effect relationship, state 'causal'; otherwise, state 'non-causal'.\n" #only gemma 
prefix_gene = "Given the following information, classify the relation between the gene pair. If there is a cause-effect relationship, state 'causal'; otherwise, state 'non-causal'.\n"
prefix_semeval = "Given the following information, classify the relation between the entity pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'.\n"

# prefix = "Classify the relation between the gene and disease pair. If there is a cause-effect relationship, say 'causal'; if not, say 'non-causal'. Use the following information as additional evidence. \n"

# Classify the relation between the gene and disease pair: TRPV6 and prostate cancer. If there's a cause-and-effect relationship, say 'causal'; if not, say 'non-causal'.
def tna_nocontext_a(test, *argv):
  pair = f"{test['e1']} and {test['e2']}"
  prompt = f"Classify the relation between the pair: {pair}. If there's a cause-and-effect relationship, say 'causal'; if not, say 'non-causal'."
  return prompt

# [Pair]: HLA-DR and BPH
# [Relation]:
def tnb_nocontext_b(test, task, model_name, *argv):
  pair = f"[Pair]: {test['e1']} and {test['e2']}\n"
  prompt = f"{prefix_mapping[task]}{pair}The relation between the pair is" #mistral
  # if 'gemma' in model_name:
  #   prompt = f"{prefix_mapping[task]}{pair}The relation between {test['e1']} and {test['e2']} is" #gemma
  # elif 'llama' in model_name:
  #   prompt = f"{prefix_mapping[task]}{pair}The relation between {test['e1']} and {test['e2']} is" #llama
  # else:
  #   prompt = f"{prefix_mapping[task]}{pair}[Relation]:" #mistral
  return prompt

# [Pair]: TRPV6 and prostate cancer
# [Context sentence]: TRPV6 and prostate cancer : cancer growth beyond the prostate correlates with increased TRPV6 Ca2+ channel expression.
# [Relation]:
def tca_context_a(test, task, *argv):
  pair = f"[Pair]: {test['e1']} and {test['e2']}\n"
  sentence = ' '.join([x for x in test['sentence'].split() if x not in remove])
  sentence = f"[Context sentence]: {sentence}\n"
  prompt = f"{prefix_mapping[task]}{pair}{sentence}[Relation]:"
  return prompt

# [Pair]: ILK and prostate cancer
# [Context sentence]: In human prostate cancer cells, knockdown of ILK expression with siRNA, or inhibition of ILK activity, results in significant inhibition of HIF-1alpha and VEGF expression.
# [Relation paths]: ['ILK', 'SLC7A1', 'prostate cancer']. 
# [Relation]:
def tcma_contextmetapath_a(test, task, model_name, n_subgraph, subgraph_mode):
  # prefix = "Given the context sentence and relation paths, classify the relation between the gene and disease pair. Respond with 'non-causal' or 'causal'.\n"
  pair = f"[Pair]: {test['e1']} and {test['e2']}\n"
  sentence = ' '.join([x for x in test['sentence'].split() if x not in remove])
  sentence = f"[Context sentence]: {sentence}\n"
  if subgraph_mode == 'random':
    random.shuffle(test['metapaths'])
    mpath = test['metapaths'][:n_subgraph]
    # mpath = sub.get_metapaths_from_json(test['e1'], test['e2'], kg_path, node_mapping, n_subgraph)
  elif subgraph_mode == 'lastk':
    mpath = test['metapaths'][-n_subgraph:]
  else: mpath = test['metapaths'][:n_subgraph]
  mpath_prefix = f"[Relation paths]: "
  mpath_context = ''
  for p in mpath:
    mpath_context = mpath_context+str(p['stops'])+'. '
  mpath_context = mpath_prefix+mpath_context+'\n'

  if 'gemma' in model_name:
    prompt = f"{prefix_mapping[task]}{pair}{sentence}{mpath_context}The relation between {test['e1']} and {test['e2']} is" #gemma
  elif 'llama' in model_name:
    prompt = f"{prefix_mapping[task]}{pair}{sentence}{mpath_context}The relation between {test['e1']} and {test['e2']} is:" #llama
  else:
    prompt = f"{prefix_mapping[task]}{pair}{sentence}{mpath_context}[Relation]:" #mistral

  return prompt

def generate_prompt(test, task, n_subgraph, template_id, subgraph_mode, model_name):
  prompt = template_mapping[template_id](test, task, model_name, n_subgraph, subgraph_mode)
  return prompt

template_mapping = {
  "tna": tna_nocontext_a,
  "tnb": tnb_nocontext_b,
  "tca": tca_context_a,
  "tcma": tcma_contextmetapath_a,
  }

prefix_mapping = {
  "ade": prefix_ade,
  "comagc": prefix_comagc,
  "gene": prefix_gene,
  "semeval": prefix_semeval,
  }

if __name__ == '__main__':
  test_prompt = generate_prompt(test, 'comagc', 1, 'tcma', 'random')
  print (test_prompt)