from torch.utils.data import Dataset
import copy
import torch

class RankingData(Dataset):
  def __init__(self, data, tokenizer, neg_num=20, include_node_labels=False, include_rel_types=False):
    self.data = data
    self.tokenizer = tokenizer
    self.neg_num = neg_num
    self.include_node_labels = include_node_labels
    self.include_rel_types = include_rel_types

  def __len__(self):
    return len(self.data)

  def __getitem__(self, item):
    item = self.data[item]
    query = f"{item['e1']} {item['e2']}"
    meta = [path for path in item['metapaths']][:self.neg_num]
    neg = []
    for path in meta:
      if self.include_node_labels:
        content = [' '.join([nl, st]) for nl, st in zip(path['nodelabels'].split(' - '), path['stops'].split(' - '))]
        content = ' - '.join(content)
        neg.append(content)
      else: neg.append(str(path['stops']))
    # neg = [str(path['stops']) for path in item['metapaths']][:self.neg_num]
    neg = neg + ['<padding_passage>'] * (self.neg_num - len(neg)) #padding if retrieved passages less than 20
    rel_score = [float(path['rel_score']) for path in item['metapaths']][:self.neg_num]
    rel_score = rel_score + [0] * (self.neg_num - len(rel_score)) #padding if retrieved passages less than 20
    passages = neg
    return [query] * len(passages), passages, rel_score

  def collate_fn(self, data):
    query, passages, rel_score = zip(*data)
    query = sum(query, [])
    passages = sum(passages, [])
    features = self.tokenizer(query, passages, padding=True, truncation=True, return_tensors="pt", max_length=500)
    features['rel_score'] = torch.Tensor(rel_score)
    return features

class RankingDataGPT(Dataset):
  def __init__(self, data, neg_num=20, include_node_labels=False, include_rel_types=False):
    self.data = data
    self.neg_num = neg_num
    self.include_node_labels = include_node_labels
    self.include_rel_types = include_rel_types

  def __len__(self):
    return len(self.data)

  def __getitem__(self, item):
    item = self.data[item]
    query = f"{item['e1']} {item['e2']}"
    meta = [path for path in item['metapaths']][:self.neg_num]
    neg = []
    for path in meta:
      if self.include_node_labels:
        content = [' '.join([nl, st]) for nl, st in zip(path['nodelabels'].split(' - '), path['stops'].split(' - '))]
        content = ' - '.join(content)
        neg.append(content)
      else: neg.append(str(path['stops']))
    return query, neg

def clean_response(response: str):
  new_response = ''
  for c in response:
    if not c.isdigit():
      new_response += ' '
    else:
      new_response += c
  new_response = new_response.strip()
  return new_response

def remove_duplicate(response):
  new_response = []
  for c in response:
    if c not in new_response:
      new_response.append(c)
  return new_response

def receive_permutation(item, permutation, rank_start=0, rank_end=100):
  response = clean_response(permutation)
  response = [int(x) - 1 for x in response.split()]
  response = remove_duplicate(response)
  cut_range = copy.deepcopy(item['metapaths'][rank_start: rank_end])
  original_rank = [tt for tt in range(len(cut_range))]
  response = [ss for ss in response if ss in original_rank]
  response = response + [tt for tt in original_rank if tt not in response]
  for j, x in enumerate(response):
    item['metapaths'][j + rank_start] = copy.deepcopy(cut_range[x])
    if 'rel_score' in item['metapaths'][j + rank_start]:
      item['metapaths'][j + rank_start]['rel_score'] = cut_range[j]['rel_score']
  return item

def write_file(rank_results, file, is_truth=True):
  print('write_file')
  # print (rank_results)
  with open(file, 'w') as f:
    for i in range(len(rank_results)):
      rank = 1
      metapaths = rank_results[i]['metapaths']
      for meta in metapaths:
        if is_truth:
          # f.write(f"{rank_results[i]['qid']} 0 {meta['pathid']} {float(meta['rel_score'])}\n")
          f.write(f"{rank_results[i]['qid']} 0 {meta['pathid']} {rank}\n")
        else:
          f.write(f"{rank_results[i]['qid']} Q0 {meta['pathid']} {rank} {meta['rel_score']} rank\n")
        rank += 1
    return True


if __name__ == '__main__':
  import json
  tokenizer = ''
  data_path = 'test_truth.jsonl'
  data = [json.loads(line) for line in open(data_path)]
  dataset = RankingData(data, tokenizer, neg_num=5, include_node_labels=True, include_rel_types=False)
  print (dataset[0])
    # (['Bax T47D', 'Bax T47D', 'Bax T47D', 'Bax T47D', 'Bax T47D'], ['BAX - PMAIP1 - Mitoxantrone - Breast cancer stage IV', 'BAX - breast cancer - Mitoxantrone - Breast cancer stage IV', 'BAX - GABARAP - Fulvestrant - Breast cancer stage IV', 'BAX - HSPA8 - Fulvestrant - Breast cancer stage IV', 'BAX - THAP11 - Fulvestrant - Breast cancer stage IV'], [4.08797, 4.19727, 4.52694, 4.60936, 4.72188])
