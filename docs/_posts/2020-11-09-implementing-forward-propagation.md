---
layout: post
title:  "Implementing Forward Propagation in NEAT"
date:   2020-11-09
tags:
  - Neural Networks
  - NEAT
  - Forward Propagation
  - Genetic Algorithm
---
For my final project in my MATH-292 class at UC Merced, I wanted to implement the [_NeuroEvolution of Augmenting Topologies_](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) (NEAT) genetic algorithm by Kenneth O. Stanley and Risto Miikulainen.
I'm not going to go into every detail about NEAT, but if you're curious about my implementation, [you can find the paper here.](https://drive.google.com/file/d/11EhKM-lHc4bFrOfuhR1WptiIW8PDjkLY/view?usp=sharing)

I originally wanted to use a popular deep learning framework like PyTorch or TensorFlow, but the main issue with them was that the structure of their neural networks are _fixed_ at runtime, and furthermore you have to create that structure yourself. The basic building blocks of these frameworks are _layers_ of neurons, and the network is _fully connected_ by default to leverage the power of matrix multiplication with GPUs.

In contrast, NEAT _evolves_ the network after each generation through mutation and crossover. Because of these mechanics, new connections and neurons are randomly generated and deleted; the network ultimately _grows larger iteratively_ with a _sparse set of connections_ (to mean that not every node in the previous layer will connect to every node in the next layer). Thus, in partnership with the time constraint I had brought upon myself due to ~~procrastination~~ underestimating the amount of research required to understand the task at hand (there already exists a [NEAT implementation that utilizes PyTorch](https://github.com/uber-research/PyTorch-NEAT), so it definitely wasn't impossible to do so), I decided to just build the network from scratch using Python.

{% include image.html url="/assets/img/NEAT versus Fixed.png" description="Left: A fully connected neural network, one that would typically be used for most deep learning frameworks like PyTorch or TensorFlow. Right: A possible network created by NEAT. Notice its irregular structure and sparse connectivity." %}

For my implementation, I wanted to use a direct encoding of the network, meaning that every single connection and node is explicitly presented. There are obviously a myriad of ways to do this, but I wanted to be as simple and straightforward as possible by representing my network as a graph - although specifically, for this implementation, it ended up as a collection of Python dictionaries. A _node_ has its type (input, output, or hidden), its values, and a list of other nodes that output to the given node; a _connection_ has a tuple (fromNode, toNode) to specify the direction of the connection, and also a weight.
