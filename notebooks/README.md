Notebooks
==============================

The purpose of these notebooks is to show how the Graph Convolutional Neural Networks were implemented from scratch and
how they compare to older and more traditional recommender systems. Models have their own individual notebook, but the
bulk of information and interest are in LightGCN and comparison.

# Summary
LightGCN is adapted from Neural Graph Collaborative Filtering (NGCF), removing
feature transformation and nonlinear activation in favor of keeping only the
essential component - neighborhood aggregation.

LightGCN vastly outperforms all other models. When compared to SVD++, LightGCN
achieves an increase in Percision@k by 29%, Recall@k by 18%, MAP by 12%, and
NDCG by 35%.

In conclusion, this demonstrates how far recommendation systems have advanced
since 2009, and how new model architectures with notable performance increases
can be developed in the span of just 1-2 years.

| Algorithm | Precision@k | Recall@k | MAP | NDCG |
| --- | --- | --- | --- | --- |
| [LightGCN](1_LightGCN.ipynb)| 0.4032 | 0.2143 | 0.1392 | 0.4603 |
| [NGCF](2_NGCF.ipynb) | 0.3573 | 0.1944 | 0.1179 | 0.4059 |
| [SVAE](3_SVAE.ipynb) | 0.3560 | 0.0929 | 0.0485 | 0.3548 |
| [SVD++](4_SVD.ipynb) | 0.1082 | 0.0386| 0.0157 | 0.1140 |
| [SVD](4_SVD.ipynb) | 0.0935 | 0.0330 | 0.0117 | 0.0927 |

# Notebook Organization

| Directory | Description |
| --- | --- |
| [EDA](0_EDA.ipynb)| A look through the MovieLens 100k dataset|
| [LightGCN](1_LightGCN.ipynb) | Implementation of LightGCN from scratch in tensorflow|
| [NGCF](2_NGCF.ipynb) | Implementation of NGCF from scratch in tensorflow |
| [SVAE](3_SVAE.ipynb) | Microsoft's implementation of SVAE |
| [SVD](4_SVD.ipynb) | Surprise implementation of SVD and SVD++|
| [comparision](5_comparison.ipynb) | Comparison of all the recommender systems above using ranking metrics (MAP, NDCG, etc...) |
