# Super Mario Bros AI Agents
## Overview
This project makes use of the Super Mario Bros Open AI Gym (https://github.com/Kautenja/gym-super-mario-bros) in order to implement several AI agents for playing SMB.

## Requirements 
In order to run this, the following packages are required: ```gym-super-mario-bros, opencv-python, argparse, multiprocessing torch matplotlib scikit-limage```.
To install, run: 
```
pip install gym-super-mario-bros opencv-python argparse multiprocessing torch matplotlib scikit-image
```
**NOTE**: ```gym-super-mario-bros``` has a dependency on ```nes-py``` which must be installed with the C-interpreter ```clang```. 
If installation of ```gym-super-mario-bros``` fails, use the flag ```CC=clang```

## Double Deep Q-Learning Agent
To run this, run the script: 
```
python deep_q_learning.py
```
The convolutional neural network used by this agent is found in ```agents/conv_neural_network.py```.
Wrapper classes used by the convolutional neural network can be found under ```util/wrappers.py```.
Code for the Double Deep Q-Learning Agent is found in ```agents/deep_q_learning_agent.py```.

## Genetic Algorithm Agent
To run the GA, run the script: 
```
python genetic_super_mario.py --num_seq <Number of Sequences> --generations <Number of generations> --world <world #> --stage <stage #>
```
The code for the GA agent is found under ```agents/genetic_sequence_agent.py```.

## Q-Learning Agent
To run Q-learning, run the script:
```
python q_learning_pixels.py
```
The code for the Q-Leaning agent is found under ```agents/q_learning_agent.py```

