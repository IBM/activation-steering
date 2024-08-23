<!-- markdownlint-disable -->

<a href="../activation_steering/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils`





---

<a href="../activation_steering/utils.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `custom_progress`

```python
custom_progress(iterable, description)
```

Create a custom progress bar for iterating over items. 



**Args:**
 
 - <b>`iterable`</b>:  The iterable to process. 
 - <b>`description`</b>:  A string describing the progress bar. 



**Yields:**
 Items from the iterable. 


---

<a href="../activation_steering/utils.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `return_default_suffixes`

```python
return_default_suffixes()
```

Return a list of default suffixes used in the CAIS representation engineering paper. 



**Returns:**
  A list of string suffixes. 


---

<a href="../activation_steering/utils.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LayerControlParams`
LayerControlParams(control: torch.Tensor | None = None, operator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = <function LayerControlParams.<lambda> at 0x7f4069cc2560>) 

<a href="../<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    control: Tensor | None = None,
    operator: Callable[[Tensor, Tensor], Tensor] = <function LayerControlParams.<lambda> at 0x7f4069cc2560>
) → None
```








---

<a href="../activation_steering/utils.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `default`

```python
default()
```

Return a default instance of LayerControlParams. 



**Returns:**
  A LayerControlParams instance with default values. 

---

<a href="../activation_steering/utils.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `<lambda>`

```python
<lambda>(current, control)
```






---

<a href="../activation_steering/utils.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ContrastivePair`
A dataclass representing a pair of contrasting strings. 



**Attributes:**
 
 - <b>`positive`</b>:  The positive string in the pair. 
 - <b>`negative`</b>:  The negative string in the pair. 

<a href="../<string>"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(positive: str, negative: str) → None
```









