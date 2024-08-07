import os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.insert(0, root_path)
import credentials
import json
import dt_utils as du

from rank_gpt import create_permutation_instruction, run_llm, receive_permutation

def process_ranking(item, lenx):
  messages = create_permutation_instruction(item=item, rank_start=0, rank_end=lenx, model_name='gpt-3.5-turbo')
  # print (f'{messages}')
  # (2) Get ChatGPT predicted permutation
  permutation = run_llm(messages, api_key=credentials.oai_key, model_name='gpt-3.5-turbo')
  print (f'{permutation=}')
  # (3) Use permutation to re-rank the passage
  return permutation

test_path = 'test_truth.jsonl'
out_path = 'test_gpt_s.jsonl'

data = [json.loads(line) for line in open(test_path)]
# dataset = du.RankingDataGPT(data, neg_num=5, include_node_labels=True)
# print (dataset[0])
with open(out_path, 'w', encoding='utf-8') as out_file:
    reranked_data = []
    for i, item in enumerate(data):
      # if i < 2:
      print (i)
      q = f"{item['e1']} {item['e2']}"
      meta = [path for path in item['metapaths']][:5]
      passages = []
      for path in meta:
        # content = [' '.join([nl, st]) for nl, st in zip(path['nodelabels'].split(' - '), path['stops'].split(' - '))]
        # content = ' - '.join(content)
        content = path['stops']
        passages.append(content)
      if len(passages) == 0:
        reranked_data.append(item)
        continue
      myitem = {}
      myitem['query'] = q
      hits = []
      for p in passages:
        dic = {}
        dic['content'] = p
        hits.append(dic)
      myitem['hits'] = hits
      response = process_ranking(myitem, len(passages))
      # response = ' > '.join([str(ss + 1) for ss in ranked])
      # print (response)
      ranked_result = du.receive_permutation(item, response, rank_start=0, rank_end=5)
      reranked_data.append(ranked_result)
      jout = json.dumps(ranked_result) + '\n'
      out_file.write(jout)

# item = {
#     'query': 'TRPV6 and prostate cancer',
#     'hits': [
#         {'content': 'TRPV6 - vagina - SQRDL - Lenalidomide - Prostate cancer metastatic '},
#         {'content': 'TRPV6 - TRAF1 - prostate cancer'},
#         {'content': 'TRPV6 - seminal vesicle - prostate cancer.'}
#     ]
# }
