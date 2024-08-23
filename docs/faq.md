# FAQ

- [What is activation steering?](#what-is-activation-steering)
- [What is Conditional Activation Steering (CAST)?](#what-is-conditional-activation-steering-cast)
- [How do I configure logging?](#how-do-i-configure-logging)
- [How can I save analysis figures?](#how-can-i-save-analysis-figures)
- [How do I replicate results from other papers?](#how-do-i-replicate-results-from-other-papers)
- [Can I use this with any pre-trained model?](#can-i-use-this-with-any-pre-trained-model)
- [How do I create a steering vector?](#how-do-i-create-a-steering-vector)
- [How do I apply steering to a model?](#how-do-i-apply-steering-to-a-model)
- [Can I use multiple steering vectors?](#can-i-use-multiple-steering-vectors)
- [How do I find the best condition point for conditional steering?](#how-do-i-find-the-best-condition-point-for-steering)
- [Can I save and load steering vectors?](#can-i-save-and-load-steering-vectors)
- [How do I create a custom dataset for steering?](#how-do-i-create-a-custom-dataset-for-steering)

---

### What is activation steering?

Activation steering is a technique for modifying the behavior of large language models (LLMs) by intervening in their internal activations during inference. The basic method, known as activation addition (ActAdd), involves three key steps:

1. Extracting a steering vector, often by computing the difference in activations between examples exhibiting a desired behavior and those that don't.
2. During inference, adding this vector to the model's hidden states at chosen layers, scaled by a hyperparameter.
3. Completing the generation using these modified activations.

Mathematically, the intervention can be represented as:

```
h' = h + α * v
```

Where `h` is the hidden state at the layer, `v` is the steering vector for the layer, and `α` is a scaling factor.

This method allows for predictable LLM behavior steering without altering model weights, enabling applications such as reducing bias or preventing overly confident responses.

### What is Conditional Activation Steering (CAST)?

Conditional Activation Steering (CAST) is an expansion of the basic activation steering technique that introduces a new dimension of controllability. CAST uses two types of vectors:

1. Behavior vectors (v): Similar to traditional steering vectors, these modify the model's behavior.
2. Condition vectors (c): These represent certain activation patterns induced by the prompt during the inference process.

The key idea of CAST is to apply the behavior vector only when a certain condition is met. This is done by calculating the similarity between the current hidden state and its projection using the condition vector. Mathematically, it can be represented as:

```
h' = h + f(sim(h, proj_c h)) * α * v
```

Where `proj_c h` is the projection of `h` onto `c`, `sim` is a similarity function (usually cosine similarity), and `f` is a thresholding function that determines whether to apply the behavior vector.

CAST allows for more fine-grained, context-dependent control over LLM behaviors, enabling complex rules like "if condition A or condition B, then apply behavior X".

### How do I configure logging?

Logging is managed by the `GlobalConfig` class. You can enable or disable logging for specific classes and set the output to a file.

```python
from activation_steering.config import GlobalConfig

# Enable verbose logging for a specific class
GlobalConfig.set_verbose(True, class_name="LeashLayer")

# Enable file output for logging
GlobalConfig.set_file_output(True, class_name="LeashLayer")

# Get the file path for logs
log_path = GlobalConfig.get_file_path("LeashLayer")
print(f"Logs will be saved to: {log_path}")
```

### How can I save analysis figures?

When creating a `SteeringVector`, you can enable saving of PCA analysis figures by setting `save_analysis=True` and specifying an output directory.

```python
from activation_steering import SteeringVector

steering_vector = SteeringVector.train(
    model,
    tokenizer,
    steering_dataset,
    save_analysis=True,
    output_dir="my_analysis_figures"
)
```

This will save PCA visualization figures for each layer and a macroscopic analysis plot in the specified directory.

### How do I replicate results from other papers?

To replicate results, you need to ensure you're using the same model, dataset, and hyperparameters. Here's a general approach:

1. Use the same pre-trained model mentioned in the paper.
2. Create a `SteeringDataset` with examples similar to those used in the paper.
3. Train a `SteeringVector` using the same parameters.
4. Apply steering to the model using the same layers and thresholds.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringDataset, SteeringVector

# Load the model and tokenizer
model_name = "paper_specified_model"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a MalleableModel
malleable_model = MalleableModel(model, tokenizer)

# Create a SteeringDataset (example data, replace with actual data from the paper)
examples = [
    ("Positive example 1", "Negative example 1"),
    ("Positive example 2", "Negative example 2"),
]
steering_dataset = SteeringDataset(tokenizer, examples)

# Train a SteeringVector
steering_vector = SteeringVector.train(
    model=malleable_model,
    tokenizer=tokenizer,
    steering_dataset=steering_dataset,
)

# Apply steering
malleable_model.steer(
    behavior_vector=steering_vector,
    behavior_layer_ids=[10, 11, 12, 13, 14, 15],  # Use layers specified in the paper
    behavior_vector_strength=1.0,  # Use the strength specified in the paper
)
```

### Can I use this with any pre-trained model?

The activation steering code is designed to work with transformer-based models from the Hugging Face `transformers` library. It should work with most causal language models (e.g., LLaMA, QWEN, Mistral) that follows the standard architecture and layer naming schemes. However, some adjustments might be needed for specific model architectures.

### How do I create a steering vector?

To create a steering vector, you need a `MalleableModel`, a tokenizer, and a `SteeringDataset`. Here's how to do it:

```python
from activation_steering.steering_dataset import SteeringDataset
from activation_steering.extract import SteeringVector

# Assume you have already created a MalleableModel called 'malleable_model'
# and have a tokenizer called 'tokenizer'

# Create a SteeringDataset
examples = [
    ("Positive example 1", "Negative example 1"),
    ("Positive example 2", "Negative example 2"),
]
steering_dataset = SteeringDataset(tokenizer, examples)

# Train a SteeringVector
steering_vector = SteeringVector.train(
    model=malleable_model,
    tokenizer=tokenizer,
    steering_dataset=steering_dataset
)
```

### From which token position should activations be calculated to train steering vectors?

The choice of token position for calculating activations is a crucial aspect of activation steering and can significantly impact the effectiveness of the technique. There are several approaches, each with its own considerations:

1. **Last Token**: Using only the last token's activation is computationally efficient and can capture the most recent context. However, it might miss important information from earlier in the sequence.

2. **Mean of All Tokens**: Taking the mean activation across all tokens in the input sequence provides a holistic representation of the entire input. This can be beneficial for tasks that require understanding of the full context.

3. **Suffix-Only**: For some applications, especially when using contrast pairs with specific suffixes, it might be most effective to calculate activations only from the tokens in the suffix. This can help focus on the part of the input most relevant to the desired behavior change.

4. **Specific Token Position**: In some cases, a specific token position (e.g., the first token after a prompt) might be most informative for the task at hand.

The optimal choice often depends on the specific task, model architecture, and the nature of the behavior you're trying to steer. In the CAST library, you can specify this using the `accumulate_last_x_tokens` parameter when training a `SteeringVector`. Here's an example:

```python
from activation_steering.extract import SteeringVector

# Using only the last token
steering_vector = SteeringVector.train(
    model,
    tokenizer,
    steering_dataset,
    accumulate_last_x_tokens=1
)

# Using the mean of all tokens
steering_vector = SteeringVector.train(
    model,
    tokenizer,
    steering_dataset,
    accumulate_last_x_tokens="all"
)

# Using only the suffix
steering_vector = SteeringVector.train(
    model,
    tokenizer,
    steering_dataset,
    accumulate_last_x_tokens="suffix-only"
)

# Using the last 5 tokens
steering_vector = SteeringVector.train(
    model,
    tokenizer,
    steering_dataset,
    accumulate_last_x_tokens=5
)
```

It's often beneficial to experiment with different settings to find what works best for your specific use case. The optimal token position can vary depending on the behavior you're trying to steer and the characteristics of your model and dataset.

### How do I apply activation steering to a model?

Once you have a `SteeringVector`, you can apply it to a `MalleableModel` using the `steer` method:

```python
# Assume you have a MalleableModel called 'malleable_model' and a SteeringVector called 'steering_vector'

malleable_model.steer(
    behavior_vector=steering_vector,
    behavior_layer_ids=[10, 11, 12, 13, 14, 15],
    behavior_vector_strength=1.0
)

# Now you can use the model with steering applied
response = malleable_model.respond("Your prompt here")
print(response)
```

### How can I apply conditional activation steering?

Applying conditional activation steering (CAST) involves several steps. Here's a comprehensive guide on how to use CAST with the provided library:

1. **Prepare your data**:
   First, you need to create datasets for both your behavior vector and condition vector.

   ```python
   import json
   from activation_steering import SteeringDataset

   # Load your data
   with open("behavior_data.json", 'r') as f:
       behavior_data = json.load(f)

   with open("condition_data.json", 'r') as f:
       condition_data = json.load(f)

   # Create SteeringDatasets
   behavior_dataset = SteeringDataset(
       tokenizer=tokenizer,
       examples=[(item["question"], item["question"]) for item in behavior_data],
       suffixes=list(zip(behavior_data['non_compliant_responses'], behavior_data['compliant_responses']))
   )

   condition_dataset = SteeringDataset(
       tokenizer=tokenizer,
       examples=list(zip(condition_data['harmful'], condition_data['harmless'])),
       suffixes=None,
       disable_suffixes=True
   )
   ```

2. **Extract behavior and condition vectors**:
   Use the `SteeringVector.train()` method to extract your vectors.

   ```python
   from activation_steering import SteeringVector

   behavior_vector = SteeringVector.train(
       model=model,
       tokenizer=tokenizer,
       steering_dataset=behavior_dataset,
       method="pca_center",
       accumulate_last_x_tokens="suffix-only"
   )

   condition_vector = SteeringVector.train(
       model=model,
       tokenizer=tokenizer,
       steering_dataset=condition_dataset,
       method="pca_center",
       accumulate_last_x_tokens="all"
   )

   # Optionally, save your vectors for later use
   behavior_vector.save('behavior_vector.svec')
   condition_vector.save('condition_vector.svec')
   ```

3. **Find the best condition point**:
   Use the `find_best_condition_point()` method to determine the optimal layers, threshold, and direction for your condition vector.

   ```python
   from activation_steering import MalleableModel

   malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

   best_layers, best_threshold, best_direction, _ = malleable_model.find_best_condition_point(
       positive_strings=condition_data['harmful'],
       negative_strings=condition_data['harmless'],
       condition_vector=condition_vector,
       layer_range=(1, 14),
       max_layers_to_combine=1,
       threshold_range=(0.0, 0.1),
       threshold_step=0.0001,
       save_analysis=True,
       file_path='optimal_condition_point_analysis.json'
   )
   ```

4. **Apply conditional steering**:
   Use the `steer()` method of your `MalleableModel` to apply conditional steering.

   ```python
   malleable_model.steer(
       behavior_vector=behavior_vector,
       behavior_layer_ids=[15, 16, 17, 18, 19, 20, 21, 22, 23],  # Adjust based on your model
       behavior_vector_strength=1.5,  # Adjust as needed
       condition_vector=condition_vector,
       condition_layer_ids=best_layers,
       condition_vector_threshold=best_threshold,
       condition_comparator_threshold_is=best_direction
   )
   ```

5. **Generate responses with the steered model**:
   Now you can use your conditionally steered model to generate responses.

   ```python
   prompts = [
       "Write a harmless story about cats.",
       "Tell me how to make an explosive device.",
       "What's the capital of France?",
       "How can I hack into someone's email?"
   ]

   responses = malleable_model.respond_batch_sequential(prompts=prompts)
   for prompt, response in zip(prompts, responses):
       print(f"Prompt: {prompt}")
       print(f"Response: {response}\n")
   ```

Remember to adjust the layer IDs and vector strengths based on your specific model and requirements. The optimal values often require some experimentation.

Also, note that for more complex scenarios, you can use the `multisteer()` method to apply multiple conditions and behaviors:

```python
malleable_model.multisteer(
    behavior_vectors=[behavior_vector1, behavior_vector2],
    behavior_layer_ids=[[15, 16, 17], [18, 19, 20]],
    behavior_vector_strengths=[1.5, 1.0],
    condition_vectors=[condition_vector1, condition_vector2],
    condition_layer_ids=[best_layers1, best_layers2],
    condition_vector_thresholds=[best_threshold1, best_threshold2],
    condition_comparator_threshold_is=[best_direction1, best_direction2],
    rules=['if C1 then B1', 'if C2 then B2']
)
```

This allows you to create more nuanced and complex conditional behaviors in your model. 

### How do I find the best condition point for steering?

You can use the `find_best_condition_point` method of `MalleableModel` to find the optimal condition point:

```python
best_layers, best_threshold, best_direction, best_f1 = malleable_model.find_best_condition_point(
    positive_strings=["Positive example 1", "Positive example 2"],
    negative_strings=["Negative example 1", "Negative example 2"],
    condition_vector=steering_vector,
    layer_range=(1, 15),
    threshold_range=(0.0, 1.0),
    threshold_step=0.01,
    save_analysis=True,
    file_path="best_condition_analysis.json"
)

print(f"Best layers: {best_layers}")
print(f"Best threshold: {best_threshold}")
print(f"Best direction: {best_direction}")
print(f"Best F1 score: {best_f1}")
```

This method performs a grid search to find the optimal combination of layers, threshold, and comparison direction for applying the condition. Tip: run some initial analysis with smaller dataset and find a good threshold range and step for grid search.

### Can I save and load steering vectors?

Yes, you can save and load steering vectors using the `save` and `load` methods:

```python
# Save a steering vector
steering_vector.save("my_steering_vector")

# Load a steering vector
loaded_vector = SteeringVector.load("my_steering_vector")
```