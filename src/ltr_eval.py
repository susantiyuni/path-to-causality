"""Approximately simulates trec_eval using pytrec_eval."""

import argparse
import os
import sys
from typing import Dict, Tuple

import pytrec_eval

def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics

def run_eval(qrel_path, run_path):
  assert os.path.exists(qrel_path)
  assert os.path.exists(run_path)

  with open(qrel_path, 'r') as f_qrel:
      qrel = pytrec_eval.parse_qrel(f_qrel)

  with open(run_path, 'r') as f_run:
      run = pytrec_eval.parse_run(f_run)
  
  all_metrics = trec_eval(qrel, run, k_values=(1, 3, 5))

  return all_metrics

# python eval.py checkpoints/1-truth.eval checkpoints/1-pred.eval
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('qrel')
    parser.add_argument('run')

    args = parser.parse_args()

    assert os.path.exists(args.qrel)
    assert os.path.exists(args.run)

    with open(args.qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(args.run, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    
    all_metrics = trec_eval(qrel, run, k_values=(1, 3, 5))
    print(all_metrics)

if __name__ == "__main__":
    sys.exit(main())