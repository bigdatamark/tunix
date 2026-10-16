# [PACKAGE DOC]
# Package tunix.rl: reinforcement-learning (RL) post-training stack for LLMs.
# Contents (subpackages/modules): algorithm configs (algorithm_config), core
# advantage/loss algorithms registered by name (algo_core, function_registry),
# shared tensor helpers (common, utils), reward aggregation (reward_manager),
# weight resharding between trainer and sampler meshes (reshard), the RL
# cluster that wires actor/critic/reference/reward models, rollout engines and
# trainers together (rl_cluster), the per-model trainer wrapper (trainer), the
# abstract RL training loop (rl_learner), algorithm learners (grpo/, ppo/),
# rollout engine backends (rollout/), inference workers (inference/), and a
# data-queue abstraction (queue/). The file itself is empty: it defines no
# symbols and has no import-time side effects.