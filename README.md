![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

# Activation Steering

👉 (Nov-2024) A few Colab demos are added.

👉 (Sep-2024) Preprint released [***Programming Refusal with Conditional Activation Steering***](https://arxiv.org/abs/2409.05907) on arXiv.

## Overview

This is a general-purpose activation steering library to (1) extract vectors and (2) steer model behavior. We release this library alongside our recent report on *Programming Refusal with Conditional Activation Steering* to provide an intuitive toolchain for activation steering efforts.

## Installation
```bash
git clone https://github.com/IBM/activation-steering

pip install -e activation-steering
```

## Activation Steering
Activation steering is a technique for influencing the behavior of language models by modifying their internal activations during inference. This library provides tools for:

- Extracting steering vectors from contrastive examples
- Applying steering vectors to modify model behavior

This part is conceptually similar to [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248), but the implementation is different.

## Conditional Activation Steering
Conditional activation steering selectively applies or withholds activation steering based on the input context. Conditional activation steering extends the activation steering framework by introducing:

- Context-dependent control capabilities through condition vectors
- Logical composition of multiple condition vectors 

Refer to our [paper](https://arxiv.org/abs/2409.05907) and [documentation](docs/quickstart.md) for detailed implementation and usage of activation steering and conditional activation steering.

## Documentation
Refer to /docs to understand this library. We recommend starting with Quick Start Tutorial as it covers most concepts that you need to get started with activation steering and conditional activation steering.

- Quick Start Tutorial (10 minutes ~ 60 minutes, depending on your hardware) 👉 [here!](docs/quickstart.md)
- FAQ 👉 [here!](docs/faq.md)

## Colab Demos

- Adding Refusal Behavior to LLaMA 3.1 8B Inst 👉 [here!](https://colab.research.google.com/drive/1IpAPMFHZW6CNrE0L16TXSvIApAK9jAFZ?usp=sharing)
- Adding CoT Behavior to Gemma 2 9B 👉 [here!](https://colab.research.google.com/drive/1dnG000syxHwOt-Z9_bpRLnBbfugI_CBh?usp=sharing)
- Making Hermes 2 Pro Conditionally Refuse Legal Instructions 👉 [here!](https://colab.research.google.com/drive/18lOzaFOK4CB_mYe9jlQbJCdHBDlhGxcQ?usp=sharing)
  
## Acknowledgement
This library builds on top of the excellent work done in the following repositories:

- [vgel/repeng](https://github.com/vgel/repeng)
- [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering)
- [nrimsky/CAA](https://github.com/nrimsky/CAA)

Some parts of the documentation for this library are generated by 

- [ml-tooling/lazydocs](https://github.com/ml-tooling/lazydocs) > lazydocs activation_steering/ --no-watermark

## Citation

```
@misc{lee2024programmingrefusalconditionalactivation,
      title={Programming Refusal with Conditional Activation Steering}, 
      author={Bruce W. Lee and Inkit Padhi and Karthikeyan Natesan Ramamurthy and Erik Miehling and Pierre Dognin and Manish Nagireddy and Amit Dhurandhar},
      year={2024},
      eprint={2409.05907},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.05907}, 
}
```
