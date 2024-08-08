# from huggingface_hub import login
# from credentials import hug_key
# login(token=hug_key)
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import utils as u

def load_models(model_name, is_use_fast=False):
  print (f'## Loading the models.. {model_name=}')
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=is_use_fast)
  tokenizer.pad_token = tokenizer.eos_token
  # tokenizer.padding_side = "right"
  # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
  model = AutoModelForCausalLM.from_pretrained(model_name)
  return model, tokenizer

neg = ['non', 'false', 'negative', 'no', 'neg', 'fal']
pos = ['caus', 'aus', 'true', 'positive', 'yes', 'pos', 'tru']

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
 
  model_name="google/gemma-7b-it"
  model, tokenizer = load_models(model_name)

  input_prompt = '''<s>[INST] Given the following information, classify the relation between the drug and side effect pair. If there is a cause-effect relationship, state 'causal'; otherwise, state 'non-causal'. 
  [Pair]: dihydrotachysterol and hypercalcemia
  [Context sentence]: Unaccountable severe hypercalcemia in a patient treated for hypoparathyroidism with dihydrotachysterol.
  [Relation paths]: ['Dihydrotachysterol', 'Paricalcitol', 'Bone pain', 'Etidronic acid', 'Hypercalcaemia of malignancy']. [/INST]
  The relation between NTR and PC-3 is 
  '''
  print (inference_with_score(model, tokenizer, input_prompt, max_new_tokens=6))


  
#     # | token | token string | logits | probability
#     # print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.4f} | {np.exp(score.numpy()):.2%}")
#     # output.scores Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam. 
#     # output.logits Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) at each generation step. 
