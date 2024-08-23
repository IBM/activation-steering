<!-- markdownlint-disable -->

<a href="../activation_steering/malleable_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `malleable_model`





---

<a href="../activation_steering/malleable_model.py#L664"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_model_layer_list`

```python
get_model_layer_list(model: MalleableModel | PreTrainedModel) → ModuleList
```

Get the list of layers from a model. 

This function handles different model architectures to retrieve their layers. 



**Args:**
 
 - <b>`model`</b>:  Either a MalleableModel or a PreTrainedModel. 



**Returns:**
 A ModuleList containing the model's layers. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the function doesn't know how to get layers for the given model type. 


---

<a href="../activation_steering/malleable_model.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MalleableModel`
MalleableModel implements conditional activation steering for language models. 

This class wraps a pre-trained language model and provides methods for applying steering vectors to modify the model's behavior conditionally. It supports both single-condition steering and multi-condition steering. 

Key features: 
- Wrap existing pre-trained models 
- Apply behavior vectors to alter model outputs 
- Condition behavior changes on input characteristics 
- Support for multi-condition steering with complex rules 



**Attributes:**
 
 - <b>`model`</b> (PreTrainedModel):  The underlying language model. 
 - <b>`tokenizer`</b> (PreTrainedTokenizerBase):  The tokenizer associated with the model. 

<a href="../activation_steering/malleable_model.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizerBase')
```

Initialize a MalleableModel instance. 

This constructor wraps a pre-trained language model and its associated tokenizer, preparing the model for conditional activation steering. It applies LeashLayer wrappers to each layer of the model, enabling fine-grained control over the model's behavior. 



**Args:**
 
 - <b>`model`</b> (PreTrainedModel):  The pre-trained language model to be wrapped. 
 - <b>`tokenizer`</b> (PreTrainedTokenizerBase):  The tokenizer associated with the model. 



**Note:**

> - The method sets the pad_token to the eos_token if not already defined. - It wraps each layer of the model with a LeashLayer for steering control. 
>

**Raises:**
 
 - <b>`AttributeError`</b>:  If the model structure is not compatible (i.e., doesn't have  'model.layers' or 'layers' attribute). 


---

#### <kbd>property</kbd> config

Get the configuration of the underlying model. 

This property provides access to the configuration object of the wrapped pre-trained model. The configuration contains model-specific parameters and settings. 



**Returns:**
 
 - <b>`PretrainedConfig`</b>:  The configuration object of the underlying model. 



**Note:**

> This is a read-only property that directly accesses the config attribute of the wrapped model. 

---

#### <kbd>property</kbd> device

Get the device on which the underlying model is located. 

This property returns the device (CPU or GPU) where the model tensors are currently allocated. It's useful for ensuring that inputs are sent to the correct device when interacting with the model. 



**Returns:**
 
 - <b>`torch.device`</b>:  The device on which the model is located (e.g., 'cpu', 
 - <b>`'cuda`</b>: 0', etc.). 



**Note:**

> The device can change if the model is moved between CPU and GPU. Always check this property before performing operations that require device-specific tensors. 



---

<a href="../activation_steering/malleable_model.py#L488"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `find_best_condition_point`

```python
find_best_condition_point(
    positive_strings: List[str],
    negative_strings: List[str],
    condition_vector: 'SteeringVector',
    layer_range: Optional[Tuple[int, int]] = None,
    max_layers_to_combine: int = 1,
    threshold_range: Tuple[float, float] = (0.0, 1.0),
    threshold_step: float = 0.01,
    save_analysis: bool = False,
    file_path: Optional[str] = None,
    condition_threshold_comparison_mode: str = 'mean'
) → Tuple[List[int], float, str, float]
```

Find the optimal condition point for steering. 



**Args:**
 
 - <b>`positive_strings`</b>:  List of strings that should trigger the condition. 
 - <b>`negative_strings`</b>:  List of strings that should not trigger the condition. 
 - <b>`condition_vector`</b>:  The steering vector representing the condition. 
 - <b>`layer_range`</b>:  Range of layers to search for the condition point. 
 - <b>`max_layers_to_combine`</b>:  Maximum number of layers to combine in the search. 
 - <b>`threshold_range`</b>:  Range of thresholds to search. 
 - <b>`threshold_step`</b>:  Step size for threshold search. 
 - <b>`save_analysis`</b>:  Whether to save the analysis results. 
 - <b>`file_path`</b>:  Path to save the analysis results. 
 - <b>`condition_threshold_comparison_mode`</b>:  Mode for comparing condition thresholds. 



**Returns:**
 A tuple containing the best layers, threshold, direction, and F1 score. 

---

<a href="../activation_steering/malleable_model.py#L632"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(*args, **kwargs)
```

Perform a forward pass through the model. 

This method delegates to the underlying model's forward method. 



**Args:**
 
 - <b>`*args`</b>:  Positional arguments to pass to the underlying model. 
 - <b>`**kwargs`</b>:  Keyword arguments to pass to the underlying model. 



**Returns:**
 The output of the underlying model's forward pass. 

---

<a href="../activation_steering/malleable_model.py#L391"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate`

```python
generate(*args, **kwargs)
```

Generate output using the underlying model. 

This method is a pass-through to the generate method of the wrapped model. It allows for text generation using the model's native generation capabilities, which may include techniques like beam search, sampling, or others depending on the underlying model architecture. 



**Args:**
 
 - <b>`*args`</b>:  Positional arguments to pass to the underlying model's generate method. 
 - <b>`**kwargs`</b>:  Keyword arguments to pass to the underlying model's generate method. 



**Returns:**
 The output generated by the underlying model's generate method. The exact return type depends on the specific model and the provided arguments. 



**Note:**

> - The behavior of this method is determined by the underlying model and the arguments passed to it. - Any steering configurations applied to the model will affect the generation process. - For detailed information on available arguments and their effects, refer to the documentation of the specific pre-trained model being used. 

---

<a href="../activation_steering/malleable_model.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multisteer`

```python
multisteer(
    behavior_vectors: List[Optional[ForwardRef('SteeringVector')]],
    behavior_layer_ids: List[List[int]],
    behavior_vector_strengths: List[float],
    condition_vectors: List[ForwardRef('SteeringVector')],
    condition_layer_ids: List[List[int]],
    condition_vector_thresholds: List[float],
    condition_comparator_threshold_is: List[str],
    rules: List[str],
    condition_threshold_comparison_modes: List[str] = None,
    use_explained_variance: bool = False,
    use_ooi_preventive_normalization: bool = False,
    apply_behavior_on_first_call: bool = True,
    **kwargs
) → None
```

Apply multiple conditional steering rules to the model. 

This method configures the model to apply multiple behavior modifications based on multiple specified conditions. It allows for complex steering scenarios with different behaviors triggered by different conditions. 



**Args:**
 
 - <b>`behavior_vectors`</b> (List[Optional[SteeringVector]]):  List of vectors representing desired behavior changes. 
 - <b>`behavior_layer_ids`</b> (List[List[int]]):  List of layers to apply each behavior vector to. 
 - <b>`behavior_vector_strengths`</b> (List[float]):  List of scaling factors for each behavior vector. 
 - <b>`condition_vectors`</b> (List[SteeringVector]):  List of vectors representing conditions for applying behaviors. 
 - <b>`condition_layer_ids`</b> (List[List[int]]):  List of layers to check each condition on. 
 - <b>`condition_vector_thresholds`</b> (List[float]):  List of thresholds for condition activations. 
 - <b>`condition_comparator_threshold_is`</b> (List[str]):  List specifying whether to activate when similarity is "larger" or "smaller" than threshold for each condition. 
 - <b>`rules`</b> (List[str]):  List of rules specifying how conditions trigger behaviors (e.g., "if C1 then B1", "if C2 or C3 then B2"). 
 - <b>`condition_threshold_comparison_modes`</b> (List[str]):  List specifying how to compare thresholds for each condition, either "mean" or "last". Default is ["mean"] * num_conditions if None. 
 - <b>`use_explained_variance`</b> (bool):  Whether to scale vectors by their explained variance. Default is False. 
 - <b>`use_ooi_preventive_normalization`</b> (bool):  Whether to use out-of-input preventive normalization. Default is False. 
 - <b>`apply_behavior_on_first_call`</b> (bool):  Whether to apply behavior vectors on the first forward call. Default is True. 
 - <b>`**kwargs`</b>:  Additional keyword arguments to pass to the LeashLayer's multisteer method. 



**Raises:**
 
 - <b>`AssertionError`</b>:  If there's a mismatch in the lengths of condition or behavior parameter lists. 



**Note:**

> - This method allows for complex steering scenarios with multiple conditions and behaviors. - Each condition can be checked on different layers, and each behavior can be applied to different layers. - The rules parameter allows for logical combinations of conditions to trigger specific behaviors. - Ensure that the lengths of all list parameters match the number of conditions or behaviors as appropriate. 

---

<a href="../activation_steering/malleable_model.py#L364"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset_leash_to_default`

```python
reset_leash_to_default() → None
```

Reset the model's steering configuration to its default state. 

This method removes all applied steering configurations, including behavior vectors and condition vectors, from all layers of the model. It resets both instance-specific and class-wide attributes of the LeashLayer wrapper. 



**Returns:**
  None 



**Note:**

> - This method should be called when you want to clear all steering configurations and return the model to its original behavior. - It's useful when you want to apply a new steering configuration from scratch or when you're done with steering and want to use the model in its default state. - This reset affects all layers of the model simultaneously. 

---

<a href="../activation_steering/malleable_model.py#L419"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `respond`

```python
respond(
    prompt,
    settings=None,
    use_chat_template=True,
    reset_after_response=True
)
```

Generate a response to a given prompt using the underlying language model. 



**Args:**
 
 - <b>`prompt`</b>:  The input prompt to generate a response for. 
 - <b>`settings`</b>:  A dictionary of generation settings. If None, default settings are used. 
 - <b>`use_chat_template`</b>:  Whether to apply the chat template to the prompt. 
 - <b>`reset_after_response`</b>:  Whether to reset the model's internal state after generating a response. 



**Returns:**
 The generated response text. 

---

<a href="../activation_steering/malleable_model.py#L467"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `respond_batch_sequential`

```python
respond_batch_sequential(prompts, settings=None, use_chat_template=True)
```





---

<a href="../activation_steering/malleable_model.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `steer`

```python
steer(
    behavior_vector: Optional[ForwardRef('SteeringVector')] = None,
    behavior_layer_ids: List[int] = [10, 11, 12, 13, 14, 15],
    behavior_vector_strength: float = 1.0,
    condition_vector: 'SteeringVector' = None,
    condition_layer_ids: List[int] = None,
    condition_vector_threshold: float = None,
    condition_comparator_threshold_is: str = 'larger',
    condition_threshold_comparison_mode: str = 'mean',
    use_explained_variance: bool = False,
    use_ooi_preventive_normalization: bool = False,
    apply_behavior_on_first_call: bool = True,
    **kwargs
) → None
```

Apply (conditional) activation steering to the model. 

This method configures the model to apply behavior modifications based on specified conditions (if given). It sets up both behavior and condition vectors across specified layers of the model. 



**Args:**
 
 - <b>`behavior_vector`</b> (Optional[SteeringVector]):  The vector representing the desired behavior change. 
 - <b>`behavior_layer_ids`</b> (List[int]):  Layers to apply the behavior vector to. Default is [10, 11, 12, 13, 14, 15]. 
 - <b>`behavior_vector_strength`</b> (float):  Scaling factor for the behavior vector. Default is 1.0. 
 - <b>`condition_vector`</b> (SteeringVector):  The vector representing the condition for applying the behavior. 
 - <b>`condition_layer_ids`</b> (List[int]):  Layers to check the condition on. 
 - <b>`condition_vector_threshold`</b> (float):  Threshold for condition activation. 
 - <b>`condition_comparator_threshold_is`</b> (str):  Whether to activate when similarity is "larger" or "smaller" than threshold. Default is "larger". 
 - <b>`condition_threshold_comparison_mode`</b> (str):  How to compare thresholds, either "mean" or "last". Default is "mean". 
 - <b>`use_explained_variance`</b> (bool):  Whether to scale vectors by their explained variance. Default is False. 
 - <b>`use_ooi_preventive_normalization`</b> (bool):  Whether to use out-of-input preventive normalization. Default is False. 
 - <b>`apply_behavior_on_first_call`</b> (bool):  Whether to apply behavior vector on the first forward call. Default is True. 
 - <b>`**kwargs`</b>:  Additional keyword arguments to pass to the LeashLayer's steer method. 



**Raises:**
 
 - <b>`ValueError`</b>:  If only one of condition_layer_ids or condition_vector is given. Omitting both is okay. 



**Note:**

> - This method updates both class and instance attributes of LeashLayer. - The behavior vector is applied only if a condition vector is not specified or if the condition is met. - Condition checking occurs only in specified layers, while behavior modification can be applied to different layers. 

---

<a href="../activation_steering/malleable_model.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `unwrap`

```python
unwrap() → PreTrainedModel
```

Remove steering modifications and return the original model. 

This method removes the LeashLayer wrappers applied to the model during initialization, returning the original, unmodified pre-trained model. 



**Returns:**
 
 - <b>`PreTrainedModel`</b>:  The original, unwrapped pre-trained model. 

Warning: After calling this method, steering functionalities (steer, reset, etc.) will no longer work as the LeashLayer instances are removed. 



**Note:**

> This method is useful when you need to access or use the original model without any steering modifications, for example, to save it or to use it with libraries that expect a standard model structure. 

---

<a href="../activation_steering/malleable_model.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `use_explained_variance`

```python
use_explained_variance(vector)
```

Apply explained variance scaling to a steering vector. 

This method scales the steering vector based on its explained variance, potentially adjusting its impact on different layers of the model. 



**Args:**
 
 - <b>`vector`</b> (SteeringVector):  The steering vector to be scaled. 



**Returns:**
 
 - <b>`numpy.ndarray`</b>:  The direction vector scaled by its explained variance. 



**Note:**

> - This method is used internally during the steering process. - It only applies scaling if the vector has an 'explained_variances' attribute. - The scaling is layer-specific, using the variance explained by each layer's principal component. 
>Warning: This method assumes that 'layer_id' is defined in the scope where it's called. Ensure that 'layer_id' is properly set before invoking this method. 


