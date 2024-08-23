<!-- markdownlint-disable -->

<a href="../activation_steering/leash_layer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `leash_layer`






---

<a href="../activation_steering/leash_layer.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LeashLayer`
A wrapper layer that implements conditional activation steering for language models. 

This layer can be applied to existing model layers to enable fine-grained control over the model's behavior through steering and conditional activation. 

Class Attributes:  condition_met: A defaultdict tracking whether conditions have been met.  forward_calls: A defaultdict counting forward passes for each layer.  condition_layers: Tracks which layers are condition layers.  behavior_layers: Tracks which layers are behavior layers.  condition_similarities: Stores condition similarities for each layer. 

<a href="../activation_steering/leash_layer.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(layer: Module, layer_id: int) → None
```

Initialize a LeashLayer. 



**Args:**
 
 - <b>`layer`</b>:  The underlying layer to be wrapped. 
 - <b>`layer_id`</b>:  The ID of this layer in the model. 




---

<a href="../activation_steering/leash_layer.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_similarity`

```python
compute_similarity(x: Tensor, y: Tensor) → float
```

Compute the cosine similarity between two tensors. 



**Args:**
 
 - <b>`x`</b>:  First tensor. 
 - <b>`y`</b>:  Second tensor. 



**Returns:**
 The cosine similarity as a float. 

---

<a href="../activation_steering/leash_layer.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(hidden_states, *args, **kwargs)
```

Perform a forward pass through this layer, applying steering if configured. 



**Args:**
 
 - <b>`hidden_states`</b>:  The input hidden states. 
 - <b>`*args`</b>:  Additional positional arguments for the underlying layer. 
 - <b>`**kwargs`</b>:  Additional keyword arguments for the underlying layer. 



**Returns:**
 The output of the underlying layer, potentially modified by steering. 

---

<a href="../activation_steering/leash_layer.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multisteer`

```python
multisteer(
    behavior_vectors: List[Tensor],
    condition_projectors: List[Tensor],
    thresholds: List[float],
    use_ooi_preventive_normalization: bool = True,
    apply_behavior_on_first_call: bool = True,
    condition_comparator_threshold_is: List[str] = ['larger'],
    condition_threshold_comparison_modes: List[str] = ['mean'],
    rules: List[str] = None,
    **kwargs
) → None
```

Configure multi-steering for this layer. 



**Args:**
 
 - <b>`behavior_vectors`</b>:  List of behavior vectors to apply. 
 - <b>`condition_projectors`</b>:  List of condition projectors to use. 
 - <b>`thresholds`</b>:  List of thresholds for condition activation. 
 - <b>`use_ooi_preventive_normalization`</b>:  Whether to use OOI preventive normalization. 
 - <b>`apply_behavior_on_first_call`</b>:  Whether to apply behavior on the first forward call. 
 - <b>`condition_comparator_threshold_is`</b>:  How to compare each condition to its threshold. 
 - <b>`condition_threshold_comparison_modes`</b>:  How to compute each condition value. 
 - <b>`rules`</b>:  List of rules for applying behaviors based on conditions. 
 - <b>`**kwargs`</b>:  Additional parameters for LayerControlParams. 

---

<a href="../activation_steering/leash_layer.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `reset_class`

```python
reset_class() → None
```

Reset the class-level attributes of LeashLayer. 

---

<a href="../activation_steering/leash_layer.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset_instance`

```python
reset_instance() → None
```

Reset this instance of LeashLayer to its default state. 

---

<a href="../activation_steering/leash_layer.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `steer`

```python
steer(
    behavior_vector: Tensor,
    condition_projector: Tensor,
    threshold: float = 0.0,
    use_ooi_preventive_normalization: bool = True,
    apply_behavior_on_first_call: bool = True,
    condition_comparator_threshold_is: str = 'larger',
    condition_threshold_comparison_mode: str = 'mean',
    **kwargs
) → None
```

Configure steering for this layer. 



**Args:**
 
 - <b>`behavior_vector`</b>:  The behavior vector to apply. 
 - <b>`condition_projector`</b>:  The condition projector to use. 
 - <b>`threshold`</b>:  The threshold for condition activation. 
 - <b>`use_ooi_preventive_normalization`</b>:  Whether to use OOI preventive normalization. 
 - <b>`apply_behavior_on_first_call`</b>:  Whether to apply behavior on the first forward call. 
 - <b>`condition_comparator_threshold_is`</b>:  How to compare the condition to the threshold. 
 - <b>`condition_threshold_comparison_mode`</b>:  How to compute the condition value. 
 - <b>`**kwargs`</b>:  Additional parameters for LayerControlParams. 


