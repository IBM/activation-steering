![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

# Activation Steering

👉 [***Programming Refusal with Conditional Activation Steering*** on arXiv](https://arxiv.org/abs/2408.09111)

## Overview

This is a general-purpose activation steering library to (1) extract vectors and (2) steer model behavior. We release this library alongside our recent report on *Programming Refusal with Conditional Activation Steering* to provide an intuitive tool chain for activation steering efforts.

## Installation
```bash
    git clone https://github.ibm.com/Bruce-Lee1/activation_steering
    
    pip install -e activation_steering
```

## Activation Steering
Activation steering is a technique for influencing the behavior of language models by modifying their internal activations during inference. This library provides tools for:

- Extracting steering vectors from contrastive examples
- Applying steering vectors to modify model behavior

## Conditional Activation Steering
Conditional activation steering selectively applies or withholds activation steering based on the input context. Conditional activation steering extends the activation steering framework by introducing:

- Context-dependent control capabilities through condition vectors
- Logical composition of multiple condition vectors 

For detailed implementation and usage of both activation steering and conditional activation steering, refer to our paper and the documentation.

## Documentation
Refer to /docs to understand this library. We recommend starting with Quick Start Tutorial as it covers most concepts that you need to get started with activation steering and conditional activation steering.

- Quick Start Tutorial (10 minutes ~ 60 minutes) 👉 [here!](docs/quickstart.md)
- FAQ 👉 [here!](docs/quickstart.md)

## Acknowledgement
This library builds on top of the excellent work done in the following repositories:

- [vgel/repeng](https://github.com/vgel/repeng)
- [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering)
- [nrimsky/CAA](https://github.com/nrimsky/CAA)

Some parts of the documentaion for this library is generated by 

- [ml-tooling/lazydocs](https://github.com/ml-tooling/lazydocs) > lazydocs activation_steering/ --no-watermark