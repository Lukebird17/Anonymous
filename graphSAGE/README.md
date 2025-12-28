### Startup Instructions for GraphSAGE

1. get social network data. run `crawl_improved.py` to get larger dataset from weibo.
2. build graph from raw data: run `step2_build_graph.py`.
3. anonymize the graph: run `step3_anonymize.py`.
4. train a GraphSAGE model to get node embeddings and perform de-anonymization attack: run `~/graphSAGE/train.py`.
5. get the attack results: run `~/graphSAGE/test.py`.
6. visualize the results: run `/generate_plots.py`.

```bash
python crawl_improved.py
python step2_build_graph.py --input data/raw/weibo_improved_data.json
python step3_anonymize.py
python graphSAGE/train.py
python graphSAGE/test.py
python generate_plots.py
```