# Supplementary materials: 
## Paths to Causality: Finding Informative Subgraphs within Knowledge Graphs for Knowledge-based Causal Discovery (KDD'25)

While traditional methods rely on observational data, knowledge-based causal discovery uses metadata (like variable names or context) to infer causality – a promising but currently unreliable approach when using LLMs alone. To improve stability and accuracy, we propose a method that combines LLMs with Knowledge Graphs.  By identifying _informative_ metapath-based subgraphs and ranking them using a Learning-to-Rank model, our approach improves zero-shot LLM prompts for causal inference, evaluated on biomedical and general-domain datasets across diverse LLMs and KGs. 

This repository includes the preprocessed [datasets](datasets/), [codes](src/), [prompt](src/zero_templatizer.py), and subgraph ranking results from the subgraph rankers. **NOTE: Part of the source code and data used for the experiments relies on proprietary libraries and therefore cannot be released.**


### Requirements
```pip install -r requirements.txt```

### Running main experiment
`bash run_kbcd.sh`

### Running (Training) Subgraph Ranker
`bash run_ltr.sh`

