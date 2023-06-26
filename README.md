# Inductive_Reasoning

The DEER dataset in [Language Models as Inductive Reasoners](https://arxiv.org/pdf/2212.10923.pdf).

In our previous arXiv version, we use a different dataset split (train 100 rules / test 100 rules), the current dataset split is (train 73 rules / test 127 rules) to better utilize the data (each rule has 6 annotated facts).

The last 22 rules in test set (id: 105~126) are obtained from gpt-3.5-turbo, while all other rules are proposed by an expert. All facts are existing texts collected from the web using search engine, after given a rule.
