## Setup

You can run this code on your own machine or on Google Colab. 

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2021/blob/master/hw1/cs285/scripts/run_hw1.ipynb)

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [infrastructure/rl_trainer.py](cs285/infrastructure/rl_trainer.py)
 - [policies/MLP_policy.py](cs285/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](cs285/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](cs285/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](cs285/infrastructure/pytorch_util.py)

Look for sections maked with `HW1` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/run_hw1.py](cs285/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](cs285/scripts/run_hw1.ipynb) (if running on Colab)
 - [agents/bc_agent.py](cs285/agents/bc_agent.py)

See the homework pdf for more details.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Section 1 (Behavior Cloning)
Command for problem 1:

```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v2.pkl
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
Command for section 1:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl \
	--video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

## Visualization the saved tensorboard event file:

You can visualize your runs using tensorboard:
```
tensorboard --logdir data
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir data/run1,data/run2,data/run3...
```

If running on Colab, you will be using the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](cs285/scripts/run_hw1.ipynb) for more details.

## Notes
Relevant lectures for this homework is [lecture 2 (all parts)](https://www.youtube.com/watch?v=HUzyjOsd2PA&list=PL_iWQOsE6TfXxKgI1GgyV1B_Xa0DxE5eH&index=5)

Imitation Learning
- take some labeled data.
- use that label data to learn some policy.
- for example, we get an image of a human driving. We label that image with an action (ex: steering wheel angle) and an observation (ex: camera attached to the car).
- then, we proceed collecting a large dataset consisting of observation-action tuples.
- And we perform supervised learning.


In this lecture we focused on two particular classes of Imitation Learning:
- Behavioral cloning:
  - It's like the above example where we copy the actions of human driving a vehicle, collect a dataset and perform supervised learning to learn a policy.
  - The problem with behavorial cloning is the following:
    - Suppose we have our training trajectory, and we learn a policy $\pi$ from this training trajectory.
    - When we run this learned policy, maybe for the first few time steps it works well. 
    - But like all models, it will make a mistake. 
    - The moment it makes this mistake, it will be in a state $s$, where $s$ is a state it has not learned before, i.e: it is not in the training trajectory.
    - Due to this "shocking" observation, the "learned" policy will take an action that is not ideal or aligned with the training trajectory.
    - So the trajectory of the policy will diverge from the training trajectory.
    - This divergence will carry on until overtime at the end of its trajectory, we see that it is not in an ideal state.
  - But... Sometimes it works.. How?
    - Well, we can modify our training data (i.e: set of training trajectories) to contain mistakes and the corresponding corrections of those mistakes.
    - This way, the policy will learn these corrections and adapt.

```
TODO:

Distribution over observation? Distributional shift problem?
```
- Dataset Aggregation (DAgger):
  - The goal is to collect training data that comes from $p_{\pi_{\theta}} (\mathbf{o_t})$ 
  - instead of $p_{data}(\mathbf{o_t})$. Let's call this dataset $\mathcal{D}_{\pi}$.
  - The algorithm is as follows:
    1. train $\pi_{\theta}(\mathbf{a_t}\, | \, \mathbf{o_t})$ from human data $\mathcal{D} = \{ \mathbf{o_1},\mathbf{a_1},\ldots,\mathbf{o_N},\mathbf{a_N} \}$
    2. run $\pi_{\theta}(\mathbf{a_t}\, | \, \mathbf{o_t})$ to get dataset $\mathcal{D}_{\pi} = \{ \mathbf{o_1}, \ldots, \mathbf{o_M} \}$
    3. ask a human (expert) to label $\mathcal{D}_{\pi}$ with actions $\mathbf{a_t}$. Yes, literaly ask a human.
    4. Aggregate (merge): $\mathcal{D} \,\cup \,\mathcal{D}_{\pi}$




