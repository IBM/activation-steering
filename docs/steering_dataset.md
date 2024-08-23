<!-- markdownlint-disable -->

<a href="../activation_steering/steering_dataset.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `steering_dataset`






---

<a href="../activation_steering/steering_dataset.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SteeringDataset`
Create a formatted dataset for steering a language model. 

This class takes a list of examples (either contrastive messages or contrastive text)  and a tokenizer, and formats the examples into a dataset of ContrastivePair objects. 

<a href="../activation_steering/steering_dataset.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    tokenizer: PreTrainedTokenizerBase,
    examples: List,
    suffixes: List[Tuple[str, str]] = None,
    disable_suffixes: bool = False,
    use_chat_template: bool = True,
    system_message: Optional[Tuple[str, str]] = None
)
```

Initialize the SteeringDataset. 



**Args:**
 
 - <b>`tokenizer`</b>:  The tokenizer used to tokenize and format the examples. 
 - <b>`examples`</b>:  A list of examples, either contrastive messages or contrastive text. 
 - <b>`suffixes`</b>:  A list of suffixes to append to the formatted dataset. If None, default suffixes will be used. 
 - <b>`disable_suffixes`</b>:  If True, no suffixes will be appended to the examples. 
 - <b>`use_chat_template`</b>:  If True, applies the chat template to the examples. 
 - <b>`system_message`</b>:  Optional system message to be included in the chat template. 




---

<a href="../activation_steering/steering_dataset.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_text`

```python
clean_text(text: str) → str
```

Clean the input text by replacing special tokens. 



**Args:**
 
 - <b>`text`</b>:  The input text to be cleaned. 



**Returns:**
 The cleaned text with special tokens replaced. 


