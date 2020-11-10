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

For my implementation, I wanted to use a direct encoding of the network, meaning that every single connection and node is explicitly presented. There are obviously a myriad of ways to do this, but I wanted to be as simple and straightforward as possible by representing my network as a graph - although specifically, for this implementation, it ended up as a collection of Python dictionaries. A _node_ has its type (input, output, or hidden), its values, and a list of other nodes that output to the given node; a _connection_ has a tuple (fromNode, toNode) to specify the direction of the connection, and also a weight. It was kind of messy if I'm honest, but it wasn't too complicated and, more importantly, the Genome class I had written worked well with the functions necessary for NEAT. Even the implementation of those NEAT functions, such as mutations, crossovers, and the delta function for speciation, weren't _too_ terrible.

No, the hardest one to implement was arguably the most simple idea in neural networks: forward propagation.

Why, you ask? For the same reason that I couldn't use TensorFlow/PyTorch: the network's structure was dynamic! Normally, I would sequentially append layers of neurons to a network in these models, and this structure would more or less define how the input flows through each hidden layer up until the output layer. However, because connections and neurons were free to randomly appear or disappear in any order, then in reality you lose the concept of individual hidden layers altogether (_all_ of the hidden nodes are considered to be the hidden layer in this case). Not to mention the further rearrangement of the connections and nodes after crossover occurs!

{% include image.html url="/assets/img/fhl v dhl.png" description="Two neural networks; in both of them, the input layer is colored blue, and the output layer is colored red. Left: a network with a set of fixed layers - it is easy to identify which neurons belong to which layer. The neurons in green are in the first hidden layer, and the neurons in purple are in the second hidden layer. Right: A network with a variety of connections, some seemingly &#34;skipping&#34; other neurons in the network. There is no real sense of different layers in this case, only a collection of hidden nodes that make up the hidden layer altogether (orange)." %}

It's important to note real quickly here what I'm using NEAT for in my case if you didn't take a look at my paper. I wanted to implement NEAT to be used with the OpenAI Gym library, a suite of Reinforcement Learning (RL) environments that developers can use to easily evaluate their RL algorithms. I know NEAT isn't _technically_ considered RL, but I was inspired by Seth Bling's [MarI/O](https://www.youtube.com/watch?v=qv6UVOQ0F44) video from a few years back to utilize this algorithm for simulations. NEAT networks will take the environment's observation as input (whether it be the raw pixels or some variables provided by the Gym environment itself), and it will output either a singular action or a set of actions to perform in the simulation (e.g. moving up/down, or moving both your left arm and right leg).

The original NEAT paper didn't mention any specific way they implemented this, so I tried to look for _generalized_ forward propagation algorithms on the internet - but all I got were a collection of articles that all implemented the same fixed topology network from scratch. I know several others have implemented this successfully, including Seth (he even posted his source code, but I couldn't really understand what he was doing in Lua), so there must have been some forward propagation algorithm out there that worked (assuming we all represented the network in the same way). Alas, I couldn't find anything that worked for me, so I had to think really hard about how I was going to feed the input through the network to produce some sort of behavior for the agent.

{% include image.html url="/assets/img/NN_fp_layered.gif" description="A small animation of how forward propagation works in typical NNs, starting with the inputs on the left and finishing with the outputs on the right. The information is fed through the system one layer at a time." %}

My approach to this implementation was creating a variant of Depth First Search (DFS), a common graph traversal algorithm we all learn in our Algorithms class. The input layer can still be filled out all at once, since the observation is ready at that point. What changes now is how we evaluate the rest of the network.

We first start by trying to evaluate all of the hidden nodes in the network; evaluating a node in this context means to a linear combination of the respective node values and weights that feed into the current node, and passing that scalar into a nonlinear activation function. There is no specific order we need to do them in, just that we need to be able to have a list of them to check through. For every node we check, we see if it is still unevaluated - if not, we check its neighbors for the values we need to do so. That same method is recursively applied to the neighbors themselves, and at a first pass, we trickle all the way down until we get to the inputs have already been evaluated. And then we work our way back up to the original hidden node by evaluating the hidden nodes preceding it - not only does it contribute towards the hidden node we are currently looking at, but it also means that we won't need to evaluate those nodes when there are other unevaluated nodes that depend on them. We do this until all of the hidden nodes have been evaluated, and then we do the same thing for the output nodes (except I use a softmax function instead).

{% include image.html url="/assets/img/NN_fp_neat.gif" description="A small animation of how forward propagation works in my NEAT algorithm. Beyond the input layer, the information is fed through the network by first recursively evaluating all of the hidden nodes, and then doing the same with the output nodes." %}

As we can see in the animation above, we still have inputs (blue nodes) on the left, outputs (red nodes) on the right, and the connections are still strictly feedforward from left to right. This time, since the hidden nodes aren't necessarily layered, they are collectively orange. Beyond the input layer, information is passed through the network by recursively evaluating first all of the hidden node, and then the output nodes by the same method. A node will check its neighbors for values, and if their neighbors also have not been evaluated yet, then those nodes will be evaluated first. The white nodes are the ones that we currently are looking at, and the green ones are its neighbors. Once a node has been evaluated, it turns grey. We do this until essentially all nodes in the network are evaluated, and from there we can get which output we want to perform by either taking the maximum output node or grabbing all of the output nodes altogether (depending on the environment).

A sample of the forward propagation implementation in Python is shown in the snippet below:
```python
# assume this snippet is a part of a Network class
# and that it's typically called as network.forward_propagation(observation)

# node_recur is the recursive helper function to forward_propagation
def node_recur(self, node):
  nodeSum = 0.0
  for n in node.neighbors:# for all nodes that connect to the current one (direction matters)
    if (self.connection[(n, node)].isExpressed): # if connection is active
      if (n.type == 'input' or n.isEvaluated): # either the node is an input or has already been evaluated (base case)
        nodeSum += n.value * connection[(n, node)].weight
      elif (not n.isEvaluated): # unevaluated (recursive case)
        nodeSum += self.node_recur(n) * self.connection[(n, node)].weight

    if (node.type == 'output'): # output node
      return np.exp(nodeSum) # to be used to calculate softmax
    else: # hidden node
      return sigmoid(nodeSum) # sigmoid function

def forward_propagation(self, observation):
  self.clear_node_values()

  # fill up the input layer
  self.inputNodes[0].value = 1.0 # the bias node
  for i in range(len(self.inputNodes)):
    self.inputNodes[i+1].value = observation[i]

  # recursively evaluate the hidden nodes
  for h in self.hiddenNodes:
    if (not h.isEvaluated):
      h.value = self.nodeRecur(h)

  # now do the same for the output nodes
  softmaxSum = 0.0
  for o in self.outputNodes:
    o.value = self.nodeRecur(o)
    softmaxSum += o.value

  # traverse through then one more time to calculate the softmax
  for o in self.outputNodes:
    o.value /= softmaxSum

  return self.outputNodes.index(max(self.outputNodes)) # return the output node with the highest value
```
Note that this is a forward propagation for a task that only requires one action at a time; if we needed to output all of the output nodes, then we'd return more than just the maximum. Furthermore, this is only for strictly feedforward networks; I tried to handle recurrent connections, but I couldn't figure out a good solution to avoid infinite recursions when trying to detect them during forward propagation. 
