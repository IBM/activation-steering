<!-- markdownlint-disable -->

<a href="../activation_steering/steering_vector.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `steering_vector`





---

<a href="../activation_steering/steering_vector.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `read_representations`

```python
read_representations(
    model: MalleableModel | PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: list[ContrastivePair],
    hidden_layer_ids: Optional[Iterable[int]] = None,
    batch_size: int = 32,
    method: Literal['pca_diff', 'pca_center'] = 'pca_center',
    save_analysis: bool = False,
    output_dir: str = 'activation_steering_figures',
    accumulate_last_x_tokens: Union[int, str] = 1,
    suffixes: List[Tuple[str, str]] = None
) → dict[int, ndarray]
```

Extract representations from the language model based on the contrast dataset. 



**Args:**
 
 - <b>`model`</b>:  The model to extract representations from. 
 - <b>`tokenizer`</b>:  The tokenizer associated with the model. 
 - <b>`inputs`</b>:  A list of ContrastivePair inputs. 
 - <b>`hidden_layer_ids`</b>:  The IDs of hidden layers to extract representations from. 
 - <b>`batch_size`</b>:  The batch size to use when processing inputs. 
 - <b>`method`</b>:  The method to use for preparing training data ("pca_diff" or "pca_center"). 
 - <b>`save_analysis`</b>:  Whether to save PCA analysis figures. 
 - <b>`output_dir`</b>:  The directory to save analysis figures to. 
 - <b>`accumulate_last_x_tokens`</b>:  How many tokens to accumulate for the hidden state. 
 - <b>`suffixes`</b>:  List of suffixes to use when accumulating hidden states. 



**Returns:**
 A dictionary mapping layer IDs to numpy arrays of directions. 


---

<a href="../activation_steering/steering_vector.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `batched_get_hiddens`

```python
batched_get_hiddens(
    model,
    tokenizer,
    inputs: list[str],
    hidden_layer_ids: list[int],
    batch_size: int,
    accumulate_last_x_tokens: Union[int, str] = 1,
    suffixes: List[Tuple[str, str]] = None
) → dict[int, ndarray]
```

Retrieve the hidden states from the specified layers of the language model for the given input strings. 



**Args:**
 
 - <b>`model`</b>:  The model to get hidden states from. 
 - <b>`tokenizer`</b>:  The tokenizer associated with the model. 
 - <b>`inputs`</b>:  A list of input strings. 
 - <b>`hidden_layer_ids`</b>:  The IDs of hidden layers to get states from. 
 - <b>`batch_size`</b>:  The batch size to use when processing inputs. 
 - <b>`accumulate_last_x_tokens`</b>:  How many tokens to accumulate for the hidden state. 
 - <b>`suffixes`</b>:  List of suffixes to use when accumulating hidden states. 



**Returns:**
 A dictionary mapping layer IDs to numpy arrays of hidden states. 


---

<a href="../activation_steering/steering_vector.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `project_onto_direction`

```python
project_onto_direction(H, direction)
```

Project a matrix H onto a direction vector. 



**Args:**
 
 - <b>`H`</b>:  The matrix to project. 
 - <b>`direction`</b>:  The direction vector to project onto. 



**Returns:**
 The projected matrix. 


---

<a href="../activation_steering/steering_vector.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_pca_figures`

```python
save_pca_figures(layer_hiddens, hidden_layer_ids, method, output_dir, inputs)
```

Save PCA analysis figures for each hidden layer and create a macroscopic x-axis layer analysis plot. 



**Args:**
 
 - <b>`layer_hiddens`</b>:  A dictionary of hidden states for each layer. 
 - <b>`hidden_layer_ids`</b>:  The IDs of hidden layers. 
 - <b>`method`</b>:  The method used for preparing training data. 
 - <b>`output_dir`</b>:  The directory to save the figures to. 
 - <b>`inputs`</b>:  The input data used for the analysis. 


---

<a href="../activation_steering/steering_vector.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SteeringVector`
A dataclass representing a steering vector used for guiding the language model. 



**Attributes:**
 
 - <b>`model_type`</b>:  The type of the model this vector is associated with. 
 - <b>`directions`</b>:  A dictionary mapping layer IDs to numpy arrays of directions. 
 - <b>`explained_variances`</b>:  A dictionary of explained variances. 

<a href="../<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model_type: str,
    directions: dict[int, ndarray],
    explained_variances: dict
) → None
```








---

<a href="../activation_steering/steering_vector.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file_path: str) → SteeringVector
```

Load a SteeringVector from a file. 



**Args:**
 
 - <b>`file_path`</b>:  The path to load the file from. If it doesn't end with '.svec',   this extension will be added. 



**Returns:**
 A new SteeringVector instance loaded from the file. 

---

<a href="../activation_steering/steering_vector.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(file_path: str)
```

Save the SteeringVector to a file. 



**Args:**
 
 - <b>`file_path`</b>:  The path to save the file to. If it doesn't end with '.svec',   this extension will be added. 

---

<a href="../activation_steering/steering_vector.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `train`

```python
train(
    model: MalleableModel | PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    steering_dataset: SteeringDataset,
    **kwargs
) → SteeringVector
```

Train a SteeringVector for a given model and tokenizer using the provided dataset. 



**Args:**
 
 - <b>`model`</b>:  The model to train the steering vector for. 
 - <b>`tokenizer`</b>:  The tokenizer associated with the model. 
 - <b>`steering_dataset`</b>:  The dataset to use for training. 
 - <b>`**kwargs`</b>:  Additional keyword arguments. 



**Returns:**
 A new SteeringVector instance. 


