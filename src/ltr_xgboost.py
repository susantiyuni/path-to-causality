import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb
import os, sys, json, pickle
import dt_utils as du
import utils as u
    
class NGramLanguageModeler(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_size):
    super(NGramLanguageModeler, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear(context_size * embedding_dim, 128)
    self.linear2 = nn.Linear(128, vocab_size)

  def forward(self, inputs):
    embeds = self.embeddings(inputs).view((1, -1))
    out = F.relu(self.linear1(embeds))
    out = self.linear2(out)
    log_probs = F.log_softmax(out, dim=1)
    return log_probs
  
  def save(self, path, mid):
    if not os.path.isdir(path): os.mkdir(path)
    model_path = os.path.join(path, mid+'_ngram_model.pkl')
    torch.save(self.state_dict(), model_path)
    print (f'Model saved to {model_path}')

  def load(self, model_path):
    map_location = None if torch.cuda.is_available() else 'cpu'
    self.load_state_dict(torch.load(model_path, map_location=map_location))
    print (f'Loading model from: {model_path}...')

def set_seed(seed):
  os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
  # random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def create_vocab(dataset, context_size):
  text = []
  for item in dataset:
    q = f"{item['e1']} {item['e2']}"
    meta = [path for path in item['metapaths']][:5]
    passages = []
    for path in meta:
      # content = [' '.join([nl, st]) for nl, st in zip(path['nodelabels'].split(' - '), path['stops'].split(' - '))]
      # content = q+' '+' - '.join(content)
      content = path['stops']
      content = q+' '+content
      passages.append(content)
    passages = ' '.join(passages)
    text.append(passages)

  text = ' '.join(text).lower().split()
  ngrams = [([text[i - j - 1] for j in range(context_size)], text[i])for i in range(context_size, len(text))]
  vocab = set(text)
  word_to_ix = {word: i for i, word in enumerate(vocab)}
  # print(f'{ngrams[:3]}, {len(vocab)}')
  return vocab, word_to_ix, ngrams

set_seed(1111)

# TASK = 'comagc'
# TASK = 'gene'
# TASK = 'ade'
TASK = 'semeval'
TEST_DATA = f'/home/yuni/kit-tud/ltr/datasets/{TASK}/16-1/test_truth.jsonl'
TRAIN_DATA = f'/home/yuni/kit-tud/ltr/datasets/{TASK}/16-1/train_full.jsonl'
SAVE_PATH ='checkpoints/embedding/'
EPOCH = 10
CONTEXT_SIZE = 2
EMBEDDING_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = [json.loads(line) for line in open(TRAIN_DATA)] + [json.loads(line) for line in open(TEST_DATA)]
vocab, word_to_ix, ngrams = create_vocab(DATASET, CONTEXT_SIZE)

loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train_embedding():
  losses = []
  best_loss = 999999999
  for epoch in tqdm(range(EPOCH), desc='Embedding training'):
    print (f"## {epoch=} ")
    total_loss = 0
    for context, target in ngrams:
      context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long).to(DEVICE)
      target_t = torch.tensor([word_to_ix[target]], dtype=torch.long).to(DEVICE)
      model.zero_grad()
      log_probs = model(context_idxs) # Step 3. Run the forward pass, getting log probabilities over next words
      loss = loss_function(log_probs, target_t)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    losses.append(total_loss)
    print (f"## {total_loss=} {best_loss=}")
    if best_loss >= total_loss:
      print (f'Save model...')
      best_loss = total_loss
      model.save(SAVE_PATH, TASK)
  return losses

def evaluate_embedding(token):
  print (f'Loading saved embedding model..')
  loaded_ngram_model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).to(DEVICE)
  MODEL_PATH = os.path.join(SAVE_PATH, TASK+'_ngram_model.pkl')
  loaded_ngram_model.load(MODEL_PATH)
  word_embedding = loaded_ngram_model.embeddings.weight([word_to_ix[token]])
  print (f'{token}: {word_embedding}')
  return word_embedding

# train_embedding()
# evaluate_embedding('loss')

def get_embedding_features(model, sentence):
  tokens = sentence.lower().split()
  vocab = {word: model.embeddings.weight[word_to_ix[word]] for word in tokens}
  embeddings = [vocab[token] for token in tokens if token in vocab]
  if len(embeddings) == 0:
    print("No embeddings found for the tokens.")
  else:
    embeddings_tensor = torch.stack([torch.tensor(embedding) for embedding in embeddings], dim=0)
    sentence_embedding = torch.mean(embeddings_tensor, dim=0).cpu().detach().numpy()
    # print("Sentence Embedding:", sentence_embedding)
  return sentence_embedding

print (f'Loading embedding model..')
embedding_model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).to(DEVICE)
MODEL_PATH = os.path.join(SAVE_PATH, TASK+'_ngram_model.pkl')
embedding_model.load(MODEL_PATH)
# print (get_embedding_features(embedding_model, 'language of work or name'))

def prepare_train_data(jsondata):
  jdata = [json.loads(line) for line in open(jsondata)]
  rels, embs, qids = [], [], []
  for item in tqdm(jdata):
    # print (item)
    q = f"{item['e1']} {item['e2']}"
    qid = item['qid']
    meta = [path for path in item['metapaths']][:5]
    # if len(meta) < 5:
    #   pad = " - "
    for r, path in enumerate(meta):
      # content = [' '.join([nl, st]) for nl, st in zip(path['nodelabels'].split(' - '), path['stops'].split(' - '))]
      # content = q+' '+' - '.join(content)
      content = path['stops']
      content = q+' '+content
      embedding = get_embedding_features(model, content)
      rels.append(r) #ranking
      embs.append(embedding)
      qids.append(int(qid))
  all_emb = np.vstack(embs)
  all_rel = np.array(rels)
  all_qid = np.array(qids)
  # print (all_emb.shape, all_rel.shape, all_qid.shape)
  return all_emb, all_rel, all_qid

def train_ranker(X, y, all_qid, fout):
  # print (f'Features:{X.shape} target:{y.shape} qids:{all_qid.shape}')
  # XGBRanker training... features:(1069, 128) target:(1069,) qids:(1069,)
  print (f'XGBRanker training... features:{X.shape} target:{y.shape} qids:{all_qid.shape}')
  ranker_model = xgb.XGBRanker(tree_method="hist",lambdarank_num_pair_per_sample=8, objective="rank:ndcg", lambdarank_pair_method="topk")
  ranker_model.fit(all_emb, all_rel, qid=all_qid)
  print (f'XGBRanker training: done!')
  pickle.dump(ranker_model, open(fout, "wb"))
  print (f'Ranker model saved! {fout}')

ranker_model_path = os.path.join(SAVE_PATH, TASK+'_ranker_model.pkl')
all_emb, all_rel, all_qid = prepare_train_data(TRAIN_DATA)

all_qid = np.sort(all_qid)
# X_train = X_train[sorted_idx, :]
# y_train = y_train[sorted_idx]

X = all_emb[all_qid, :]
y = all_rel[all_qid]
train_ranker(X, y, all_qid, ranker_model_path)

print ('Testing ranking... ')
xgb_model_loaded = pickle.load(open(ranker_model_path, "rb"))
print (f'Model loaded! {ranker_model_path}')

checkpoint_path = f"checkpoints/{TASK}/16-1/xgboost"
u.folder_check(checkpoint_path)
jdata = [json.loads(line) for line in open(TEST_DATA)]
with open(checkpoint_path+'/test_pred_xgboost.jsonl', 'w', encoding='utf-8') as out_file:
  reranked_data = []
  for item in tqdm(jdata):
    # print (item)
    q = f"{item['e1']} {item['e2']}"
    qid = item['qid']
    meta = [path for path in item['metapaths']][:5]
    embs = []
    for r, path in enumerate(meta):
      # content = [' '.join([nl, st]) for nl, st in zip(path['nodelabels'].split(' - '), path['stops'].split(' - '))]
      # content = q+' '+' - '.join(content)
      content = path['stops']
      content = q+' '+content
      embedding = get_embedding_features(embedding_model, content)
      embs.append(embedding)
    all_emb = np.vstack(embs)
    scores = xgb_model_loaded.predict(all_emb)
    # normalized_scores = [float(score) for score in scores]
    sorted_idx = np.argsort(scores)[::-1]
    ranked_scores = scores[sorted_idx]
    response = ' > '.join([str(ss + 1) for ss in sorted_idx])
    print (scores, ranked_scores, response)
    ranked_result = du.receive_permutation(item, response, rank_start=0, rank_end=5)
    reranked_data.append(ranked_result)
    # print (ranked_result)
    jout = json.dumps(ranked_result) + '\n'
    out_file.write(jout)
    # sys.exit()

print (f'Finished! {checkpoint_path} ')

# xgb_model_loaded.predict(test) == xgb_model.predict(test)[0]

# scores = ranker.predict(X)
# scores = xgb_model_loaded.predict(X)
# # sorted_idx = np.argsort(scores)[::-1]
# # # Sort the relevance scores from most relevant to least relevant
# # scores = scores[sorted_idx]
# print (scores, scores.shape)