# DT-MW method

DT-MW uses two decision timescales for ride-hailing dispatching and repositioning.

1. The **Manager** reads scenario context, such as severe weather or a demand surge. It emits a strategy instruction, a prior over the service and fairness dual variables, and minimum service and fairness thresholds.
2. The **Dual controller** adjusts the prior using observed order response rate and driver-income fairness. The corrected dual variables determine the weights on GMV, order response, and fairness in dynamic reward shaping.
3. The **Worker** uses a MAPPO policy at each dispatch step. Semantic state augmentation adds the manager's instruction embedding to the node observation. Intrinsic semantic alignment uses a contrastive head to align worker state embeddings with manager instructions.

| Component | Implementation |
| --- | --- |
| Scenario and online manager backends | [llm/unified_manager.py](../llm/unified_manager.py) |
| Feedback and reward weights | [llm/fusion_manager.py](../llm/fusion_manager.py) |
| Dynamic reward shaping | [llm/reward_shaper.py](../llm/reward_shaper.py) |
| MAPPO worker and contrastive head | [llm_algo/mappo_llm.py](../llm_algo/mappo_llm.py) |
| Rollout, training, and metrics | [run/train.py](../run/train.py), [run/eval.py](../run/eval.py) |
