<!-- markdownlint-disable -->

<a href="../activation_steering/config.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `config`





---

<a href="../activation_steering/config.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `log`

```python
log(message: str, style: str = None, class_name: str = 'global')
```

Log a message to the console and/or file based on the current configuration. 



**Args:**
 
 - <b>`message`</b> (str):  The message to log. 
 - <b>`style`</b> (str, optional):  The style to apply to the console output. 
 - <b>`class_name`</b> (str):  The class name associated with the log message. Defaults to "global". 


---

<a href="../activation_steering/config.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LogConfig`
Configuration class for logging settings. 



**Attributes:**
 
 - <b>`enabled`</b> (bool):  Whether logging is enabled. 
 - <b>`file_output`</b> (bool):  Whether to output logs to a file. 
 - <b>`file_path`</b> (str):  Path to the log file. 

<a href="../activation_steering/config.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(enabled=True)
```

Initialize a LogConfig instance. 



**Args:**
 
 - <b>`enabled`</b> (bool):  Initial enabled state for logging. 





---

<a href="../activation_steering/config.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GlobalConfig`
Singleton class for global configuration settings. 

Class Attributes:  console (Console): Rich console instance for pretty printing.  log_configs (dict): Dictionary of LogConfig instances for different classes.  log_directory (str): Directory for storing log files. 




---

<a href="../activation_steering/config.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `get_file_path`

```python
get_file_path(class_name: str)
```

Get the log file path for a specific class. 



**Args:**
 
 - <b>`class_name`</b> (str):  The class name to get the file path for. 



**Returns:**
 
 - <b>`str`</b>:  The path to the log file. 

---

<a href="../activation_steering/config.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `initialize_log_files`

```python
initialize_log_files()
```

Initialize log files for all configured classes. 

---

<a href="../activation_steering/config.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `is_verbose`

```python
is_verbose(class_name: str = 'global')
```

Check if verbose logging is enabled for a specific class. 



**Args:**
 
 - <b>`class_name`</b> (str):  The class name to check. Defaults to "global". 



**Returns:**
 
 - <b>`bool`</b>:  True if verbose logging is enabled, False otherwise. 

---

<a href="../activation_steering/config.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `set_file_output`

```python
set_file_output(enabled: bool, class_name: str = 'global')
```

Set whether to output logs to a file for a specific class or globally. 



**Args:**
 
 - <b>`enabled`</b> (bool):  Whether to enable file output. 
 - <b>`class_name`</b> (str):  The class name to set file output for. Defaults to "global". 

---

<a href="../activation_steering/config.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `set_verbose`

```python
set_verbose(verbose: bool, class_name: str = 'global')
```

Set the verbose state for a specific class or globally. 



**Args:**
 
 - <b>`verbose`</b> (bool):  Whether to enable verbose logging. 
 - <b>`class_name`</b> (str):  The class name to set verbose for. Defaults to "global". 

---

<a href="../activation_steering/config.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `should_log_to_file`

```python
should_log_to_file(class_name: str)
```

Check if logging to a file is enabled for a specific class. 



**Args:**
 
 - <b>`class_name`</b> (str):  The class name to check. 



**Returns:**
 
 - <b>`bool`</b>:  True if file logging is enabled, False otherwise. 


