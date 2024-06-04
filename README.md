# Reinforcement learning and regret bounds for admission control

This directory contains the source code related to UCRL-AC of the paper: Reinforcement Learning and Regret Bounds for Admission Control.
The code of the algorithms UCRL-AC is compared with can be found at https://gitlab.inria.fr/omaillar/article-companion/-/tree/master/2020-UCRL3.


## Run UCRL-AC
Run the following code in a terminal:
```
python3 experiments.py SERVICE_RATE QUEUE_SIZE   
```
SERVICE_RATE and BUFFER_SIZE must be replaced with their desired values.

## Plot experimental results
plot.py contains the code to compute the mean regret and the bounds of the 95% confidence interval.

## Structure of the code
```
agent.py                types of agent that have been used
buffers.py              generate sequence of arrivals to increase speed and reproducibility
experiments.py          run an experiment for a given number of classes, service rate and buffer size
plot.ipy                notebook used to produce the figures
system.py               queuing system
test.py                 code testing some functions.
utils.py                auxiliary code
```
