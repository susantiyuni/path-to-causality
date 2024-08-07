# from huggingface_hub import login
# from credentials import hug_key
# login(token=hug_key)
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import utils as u

neg = ['non', 'false', 'negative', 'no', 'neg', 'fal']
pos = ['caus', 'aus', 'true', 'positive', 'yes', 'pos', 'tru']

def load_models(model_name, is_use_fast=False):
  print (f'## Loading the models.. {model_name=}')
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=is_use_fast)
  tokenizer.pad_token = tokenizer.eos_token
  # tokenizer.padding_side = "right"
  # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
  model = AutoModelForCausalLM.from_pretrained(model_name)
  return model, tokenizer

def inference_with_score(model, tokenizer, input_prompt, max_new_tokens=50):
  print (f'## Inference..')
  inputs= tokenizer(input_prompt, return_tensors="pt")
  output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)
  transition_scores = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
  # print(f"{transition_scores=} {transition_scores.shape}")
  # print (f"{output.sequences=} {output.sequences.shape}")
  # generated_sequence = tokenizer.batch_decode(output.sequences, skip_special_tokens= True)[0]
  # print(f"{generated_sequence=}")
  input_length = inputs.input_ids.shape[1]
  generated_tokens = output.sequences[:, input_length:]
  out_llm = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
  out_verb = u.verbalizer(out_llm)
  # all_score = [[x, y] for x, y in zip(generated_tokens[0], transition_scores[0]]
  for tok, score in zip(generated_tokens[0], transition_scores[0]):
    print(f"| {tok:5d} | {repr(tokenizer.decode(tok)):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
    tok = tokenizer.decode(tok).lower().strip()
    if tok in neg:
      print (f"NEG: {score.numpy():.4f} | {repr(tok)}")
      return out_verb, out_llm, score.numpy()
    elif tok in pos:
      print (f"POS: {score.numpy():.4f} | {repr(tok)}")
      return out_verb, out_llm, score.numpy()
  return out_verb, out_llm, transition_scores[0][0].numpy()

def inference(model, tokenizer, input_prompt, max_new_tokens=7):
  print (f'## Inference..')
  # inputs= tokenizer.encode(input_prompt, return_tensors="pt") --> only return input_ids
  # output = model.generate(inputs)
  # inputs= tokenizer(input_prompt, return_tensors="pt").to("cuda")
  inputs= tokenizer(input_prompt, return_tensors="pt")
  output = model.generate(
          input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)[:, inputs["input_ids"].shape[1]:]
  output_decode = tokenizer.decode(output[0], skip_special_tokens=True)
  # print (f'{output_decode=}')
  return output_decode

def elaborate(model, tokenizer, input_prompt, max_new_tokens=150):
  print (f'## Generate description..')
  model_inputs = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False)
  generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
  decoded = tokenizer.batch_decode(generated_ids)
  # print(decoded[0])
  output = decoded[0].split("[/INST]")[1]
  return output

if __name__ == '__main__':
  u.set_seed(1234)
  # model_name="BioMistral/BioMistral-7B"
  # model_name="chaoyi-wu/PMC_LLAMA_7B"
  # model_name="medalpaca/medalpaca-7b"
  # model_name="epfl-llm/meditron-7b"
  # model_name="PharMolix/BioMedGPT-LM-7B"
  # model_name="dmis-lab/meerkat-7b-v1.0"

  model_name = "mistralai/Mistral-7B-Instruct-v0.1"
  # model_name="BioMistral/BioMistral-7B"
  model_name="google/gemma-7b-it"
  model, tokenizer = load_models(model_name)

  # input_prompt = '''<s>[INST] Given the following as additional information, classify the relation between the gene and disease pair. If there is a cause-effect relationship, state 'causal'; otherwise, state 'non-causal'.
  # [Pair]: NTR and PC-3
  # [Context sentence]: Transfection of the PC-3 prostate cell line with a dominant-negative form of p75( NTR ) before DIM treatment significantly rescued cell survival demonstrating a cause and effect relationship between DIM induction of p75(NTR) levels and inhibition of survival.
  # [Relation paths]: ['NKTR', 'adrenal gland', 'SQRDL', 'Lenalidomide', 'Prostate cancer metastatic']. [/INST]
  # The relation between NTR and PC-3 is
  # '''

  # input_prompt = ''' <s>[INST] I will provide you with 5 relation paths, each indicated by number identifier []. \nRank the relation paths based on their relevance to support the statement: There is a causal relationship between MMP-9 and PC-3
  # [1] ['MMP9', 'CREBBP', 'hematologic cancer', 'Lenalidomide', 'Prostate cancer metastatic']
  # [2] ['MMP9', 'JUND', 'hematologic cancer', 'Lenalidomide', 'Prostate cancer metastatic']
  # [3] ['MMP9', 'hematopoietic system', 'RRP8', 'Lenalidomide', 'Prostate cancer metastatic']
  # [4] ['MMP9', 'female reproductive system', 'RRP8', 'Lenalidomide', 'Prostate cancer metastatic']
  # [5] ['MMP9', 'Captopril', 'Pruritus', 'Lenalidomide', 'Prostate cancer metastatic']
  # The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results. [/INST]
  # '''
  
  # input_prompt = '''
  # Given a statement "There is a non-causal relation between NF-κB and	prostate cancer", which of the following two paths is more relevant to the statement?
  # A: ['NFIB', 'RPN1', 'prostate cancer'] ## not relevant
  # B: ['NFIB', 'CBR1', 'prostate cancer'] ## relevant
  # Answer only with 'A' or 'B'.
  # '''

  # input_prompt = '''<s>[INST] Pair: NF-κB and prostate cancer. Relation path: ['NFIB', 'RPN1', 'prostate cancer']. Is the relation path relevant for predicting the causal relation between the pair? Answer with 'Yes' or 'No'. [/INST] Answer: '''

  # input_prompt = '''<s>[INST] Statement: (MMP-9, causal, PC-3). Relation path: ['MMP9', 'CREBBP', 'hematologic cancer', 'Lenalidomide', 'Prostate cancer metastatic']. Does the relation path support the statement? Answer with 'Yes' or 'No'. [/INST] Answer: '''

  # print (inference(model, tokenizer, input_prompt, 6))
  input_prompt = '''<s>[INST] Given the following information, classify the relation between the drug and side effect pair. If there is a cause-effect relationship, state 'causal'; otherwise, state 'non-causal'. 
  [Pair]: dihydrotachysterol and hypercalcemia
  [Context sentence]: Unaccountable severe hypercalcemia in a patient treated for hypoparathyroidism with dihydrotachysterol.
  [Relation paths]: ['Dihydrotachysterol', 'Paricalcitol', 'Bone pain', 'Etidronic acid', 'Hypercalcaemia of malignancy']. [/INST]
  The relation between NTR and PC-3 is 
  '''
  print (inference_with_score(model, tokenizer, input_prompt, max_new_tokens=6))



# class StoppingCriteriaSub(StoppingCriteria):
#   def __init__(self, stops = [], encounters=1):
#     super().__init__()
#     self.stops = [stop for stop in stops]

#   def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#     for stop in self.stops:
#       if torch.all((stop == input_ids[0][-len(stop):])).item():
#         return True
#     return False

# stop_words = ["\n"]
# stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
# stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

# def inference_with_scorex(model, tokenizer, input_prompt, max_new_tokens=50):
#   print (f'## Inference..')
#   inputs= tokenizer(input_prompt, return_tensors="pt")
#   output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)
#   transition_scores = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
#   # print(f"{transition_scores=} {transition_scores.shape}")
#   # print (f"{output.sequences=} {output.sequences.shape}")
#   # generated_sequence = tokenizer.batch_decode(output.sequences, skip_special_tokens= True)[0]
#   # print(f"{generated_sequence=}")

#   input_length = inputs.input_ids.shape[1]
#   generated_tokens = output.sequences[:, input_length:]
#   out_llm = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
#   out_verb = u.verbalizer(out_llm)
#   for tok, score in zip(generated_tokens[0], transition_scores[0]):
#     print(f"| {tok:5d} | {repr(tokenizer.decode(tok)):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
#     tok = tokenizer.decode(tok).lower()
#     if tok in neg:
#       print (f"NEG: {score.numpy():.4f} | {repr(tok)}")
#       return out_verb, out_llm, score.numpy()
#     elif tok in pos:
#       print (f"POS: {score.numpy():.4f} | {repr(tok)}")
#       return out_verb, out_llm, score.numpy()
#   return out_verb, out_llm, transition_scores[0][0]
#     # else: 
#     #   print (f"ELSE/NEG: {score.numpy():.4f} | {repr(tok)}")
#     #   return out_verb, out_llm, score.numpy()

#     # dic.append([relscore, tok])
#     # if out_verb == 1:
#     #   relscore = - 1 / score.numpy()
#     # else: relscore = 1 / score.numpy()
#     # if any(v in tok for v in neg):
#     #   print (f"NEG: {score.numpy():.4f} | {repr(tok)}")
#     #   relscore = 1 / score.numpy()
#     #   return out_verb, out_llm, relscore
#     # elif any(v in tok for v in pos):
#     #   print (f"POS: {score.numpy():.4f} | {repr(tok)}")
#     #   relscore = - 1 / score.numpy()
#     #   return out_verb, out_llm, relscore
#     # else: 
#     #   print (f"ELSE/NEG: {score.numpy():.4f} | {repr(tok)}")
#     #   relscore = - 1 / score.numpy()
#     #   return out_verb, out_llm, relscore
#   # return out_verb, out_llm, dic
  
#     # | token | token string | logits | probability
#     # print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
#     # output.scores Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam. 
#     # output.logits Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. 
