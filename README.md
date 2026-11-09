# Machine Learning

My practice code from a self-study course on neural networks. It starts with a single neuron worked out by hand and ends with models and agents that keep learning across separate runs of a program, using checkpoints, continual learning and reinforcement learning.

## Where this came from

I used an AI assistant to write the course. That covers the written guide (kept in my Google Drive, not in this repository), the 22 test scripts, the data module and the reference solutions to the 14 tasks, all of it AI-generated. My own code goes in [`my_work/`](my_work/): my answers to the tasks, which I then compare with the reference solutions.

## What's here

| Path | What it is | Written by |
|---|---|---|
| `t01_neuron.py` … `t22_hopfield.py` | 22 short experiments, each making one idea visible. Each prints its key numbers and saves a figure. | AI (course material) |
| `mlguide_data.py` | Generates all the data locally (2-D toy datasets and 16×16 synthetic digit images), plus shared helpers. Nothing is downloaded. | AI (course material) |
| `solutions/` | Reference solutions to the 14 tasks. | AI (course material) |
| `my_work/` | My own solutions to the tasks. | Me |

## Setup

You need Python 3.10 or newer. The CPU build of PyTorch is enough.

```bash
pip install -r requirements.txt
```

## Running

Run any script from this folder:

```bash
python t01_neuron.py
python solutions/task_04_backprop.py
```

Use plain `python`, not `python -P`, because the scripts import `mlguide_data.py` from this folder. Figures and checkpoints are saved in `results/` and generated data is cached in `data_cache/`; Git ignores both. Each test takes under two minutes on a laptop CPU, and some longer ones accept `--quick` for a shorter run (see the docstring at the top of each file).

## The tests

| # | File | What it shows | Guide section |
|---|---|---|---|
| 1 | `t01_neuron.py` | A single neuron and its decision line | 1.2 |
| 2 | `t02_bias.py` | Why biases matter | 1.2 |
| 3 | `t03_nonlinearity.py` | Why non-linearity matters | 1.2 |
| 4 | `t04_init.py` | Starting weights: zeros, too small, too big, right | 1.4 |
| 5 | `t05_learning_rate.py` | The learning rate | 1.3 |
| 6 | `t06_backprop_check.py` | Backpropagation by hand and gradient checking | 1.3 |
| 7 | `t07_width.py` | Width | 2.2 |
| 8 | `t08_depth.py` | Depth, plain vs residual | 2.2 |
| 9 | `t09_overfitting.py` | Overfitting and regularisation | 1.4 |
| 10 | `t10_cnn_vs_mlp.py` | CNN vs MLP on shifted digits | 3.1 |
| 11 | `t11_memory_length.py` | How long RNNs, LSTMs and GRUs remember | 3.2 |
| 12 | `t12_attention.py` | Attention as lookup | 3.3 |
| 13 | `t13_spiking.py` | A spiking neuron and a small spiking network | 3.4 |
| 14 | `t14_checkpoint.py` | Saving and resuming exactly | 4.1 |
| 15 | `t15_forgetting.py` | Catastrophic forgetting | 4.3 |
| 16 | `t16_replay.py` | Replay | 4.3 |
| 17 | `t17_ewc.py` | Elastic weight consolidation | 4.3 |
| 18 | `t18_bandits.py` | Bandits | 4.4 |
| 19 | `t19_q_learning.py` | Q-learning, then a neural Q-network | 4.4 |
| 20 | `t20_plasticity.py` | Loss of plasticity | 4.3 |
| 21 | `t21_count_params.py` | Counting parameters | 2.1 |
| 22 | `t22_hopfield.py` | A Hopfield network remembers | 4.2 |

## The tasks

The tasks build on each other, so they are meant to be done in order, about 45 hours in all.

| # | Task | Reference solution |
|---|---|---|
| 1 | A neuron worked out by hand (the solution checks the pen-and-paper answers) | `solutions/task_01_neuron_by_hand.py` |
| 2 | A neuron and a layer in NumPy, checked against PyTorch | `solutions/task_02_neuron_layer.py` |
| 3 | Forward pass and parameter count of a small MLP | `solutions/task_03_mlp_forward.py` |
| 4 | Backpropagation written by hand, with gradient checks | `solutions/task_04_backprop.py` |
| 5 | Training the NumPy network on three-arm spirals | `solutions/task_05_train_numpy.py` |
| 6 | The same network in PyTorch, checked against NumPy | `solutions/task_06_pytorch.py` |
| 7 | Designing networks: parameters, compute and memory | `solutions/task_07_design.py` |
| 8 | An LSTM cell from its equations, checked against `nn.LSTM` | `solutions/task_08_lstm_cell.py` |
| 9 | Scaled dot-product and multi-head attention from scratch | `solutions/task_09_attention.py` |
| 10 | A training loop that saves and resumes exactly | `solutions/task_10_checkpoint.py` |
| 11 | Continual learning over five tasks | `solutions/task_11_continual.py` |
| 12 | An agent that keeps learning across separate runs | `solutions/task_12_agent.py` |
| 13 | EWC on permuted digits: Kirkpatrick et al. (2017) in miniature | `solutions/task_13_ewc_paper.py` |
| 14 | Capstone: a model that keeps learning, session after session | `solutions/task_14_capstone.py` |

The section numbers in the tests table (1.2, 3.1 and so on) refer to the written guide.

## Licence

MIT. See [LICENSE](LICENSE).
