# FedSC: Federated Learning with Semantic-Aware Collaboration

This is an official implementation of the following paper:
> Huan Wang, Haoran Li, Huaming Chen, Jun Yan, Jiahua Shi, Jun Shen. *"FedSC: Federated Learning with Semantic-Aware Collaboration"*. ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), KDD 2025.
---

**Abstract:** Federated learning (FL) aims to train models collaboratively across clients without sharing data for privacy-preserving. However, one major challenge is the data heterogeneity issue, which refers to the biased labeling preferences at multiple clients. A number of existing FL methods attempt to tackle data heterogeneity locally (e.g., regularizing local models) or globally (e.g., fine-tuning global model), often neglecting inherent semantic information contained in each client. To explore the possibility of using intra-client semantically meaningful knowledge in handling data heterogeneity, in this paper, we propose Federated Learning with Semantic-Aware Collaboration (FedSC) to capture client-specific and class-relevant knowledge across heterogeneous clients. The core idea of FedSC is to construct relational prototypes and consistent prototypes at semantic-level, aiming to provide fruitful class underlying knowledge and stable convergence signals in a prototype-wise collaborative way. On the one hand, FedSC introduces an inter-contrastive learning strategy to bring instance-level embeddings closer to relational prototypes with the same semantics and away from distinct classes. On the other hand, FedSC devises consistent prototypes via a discrepancy aggregation manner, as a regularization penalty to constrain the optimization region of the local model. Moreover, a theoretical analysis for FedSC is provided to ensure a convergence guarantee. Experimental results on various challenging scenarios demonstrate the effectiveness of FedSC and the efficiency of crucial components.

---

Here is an example to run FedSC on CIFAR-10 with noniid_factor=0.05 & imb_factor=0.1:


```python
python3 main_fedsc.py --data_name cifar10 --num_classes 10 --non_iid_alpha 0.05 --imb_factor 0.1
```