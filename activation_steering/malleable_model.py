import typing, os
from typing import List, Union, Tuple, Optional
from itertools import combinations
from collections import defaultdict
import json

import torch
from sklearn.metrics import f1_score
import numpy as np
from transformers import PretrainedConfig, PreTrainedModel

from activation_steering.leash_layer import LeashLayer
from activation_steering.utils import custom_progress
from rich.progress import track
from rich.table import Table
from activation_steering.config import log



if typing.TYPE_CHECKING:
    from .extract import SteeringVector


class MalleableModel(torch.nn.Module):
    """
    MalleableModel implements conditional activation steering for language models.

    This class wraps a pre-trained language model and provides methods for applying
    steering vectors to modify the model's behavior conditionally. It supports both
    single-condition steering and multi-condition steering.

    Key features:
    - Wrap existing pre-trained models
    - Apply behavior vectors to alter model outputs
    - Condition behavior changes on input characteristics
    - Support for multi-condition steering with complex rules

    Attributes:
        model (PreTrainedModel): The underlying language model.
        tokenizer (PreTrainedTokenizerBase): The tokenizer associated with the model.
    """
    def __init__(self, model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizerBase'):
        """
        Initialize a MalleableModel instance.

        This constructor wraps a pre-trained language model and its associated tokenizer,
        preparing the model for conditional activation steering. It applies LeashLayer
        wrappers to each layer of the model, enabling fine-grained control over the
        model's behavior.

        Args:
            model (PreTrainedModel): The pre-trained language model to be wrapped.
            tokenizer (PreTrainedTokenizerBase): The tokenizer associated with the model.

        Note:
            - The method sets the pad_token to the eos_token if not already defined.
            - It wraps each layer of the model with a LeashLayer for steering control.

        Raises:
            AttributeError: If the model structure is not compatible (i.e., doesn't have
                            'model.layers' or 'layers' attribute).
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token # Most LLMs don't have a pad token by default

        # Get the actual layers
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers

        # Wrap each layer with LeashLayer in place
        for i in range(len(layers)):
            if not isinstance(layers[i], LeashLayer):
                layers[i] = LeashLayer(layers[i], i)

        log(f"... The target model type is [cyan]{model.config.model_type}[/cyan].", style="magenta", class_name="MalleableModel")

    @property
    def config(self) -> PretrainedConfig:
        """
        Get the configuration of the underlying model.

        This property provides access to the configuration object of the wrapped
        pre-trained model. The configuration contains model-specific parameters
        and settings.

        Returns:
            PretrainedConfig: The configuration object of the underlying model.

        Note:
            This is a read-only property that directly accesses the config
            attribute of the wrapped model.
        """
        return self.model.config

    @property
    def device(self) -> torch.device:
        """
        Get the device on which the underlying model is located.

        This property returns the device (CPU or GPU) where the model tensors
        are currently allocated. It's useful for ensuring that inputs are sent
        to the correct device when interacting with the model.

        Returns:
            torch.device: The device on which the model is located (e.g., 'cpu',
                        'cuda:0', etc.).

        Note:
            The device can change if the model is moved between CPU and GPU.
            Always check this property before performing operations that require
            device-specific tensors.
        """
        return self.model.device

    def unwrap(self) -> PreTrainedModel:
        """
        Remove steering modifications and return the original model.

        This method removes the LeashLayer wrappers applied to the model during
        initialization, returning the original, unmodified pre-trained model.

        Returns:
            PreTrainedModel: The original, unwrapped pre-trained model.

        Warning:
            After calling this method, steering functionalities (steer, reset, etc.)
            will no longer work as the LeashLayer instances are removed.

        Note:
            This method is useful when you need to access or use the original
            model without any steering modifications, for example, to save it
            or to use it with libraries that expect a standard model structure.
        """
        layers = get_model_layer_list(self.model)
        for layer_id in list(range(len(layers))):
            if isinstance(layers[layer_id], LeashLayer):
                layers[layer_id] = layers[layer_id].layer
        return self.model

    def use_explained_variance(vector):
        """
        Apply explained variance scaling to a steering vector.

        This method scales the steering vector based on its explained variance,
        potentially adjusting its impact on different layers of the model.

        Args:
            vector (SteeringVector): The steering vector to be scaled.

        Returns:
            numpy.ndarray: The direction vector scaled by its explained variance.

        Note:
            - This method is used internally during the steering process.
            - It only applies scaling if the vector has an 'explained_variances' attribute.
            - The scaling is layer-specific, using the variance explained by each layer's
            principal component.

        Warning:
            This method assumes that 'layer_id' is defined in the scope where it's called.
            Ensure that 'layer_id' is properly set before invoking this method.
        """
        if hasattr(vector, 'explained_variances'):
            variance_scale = vector.explained_variances.get(layer_id, 1)
            direction = direction * variance_scale
        return direction
        
    def steer(self, behavior_vector: Optional["SteeringVector"] = None, behavior_layer_ids: List[int] = [10, 11, 12, 13, 14, 15], behavior_vector_strength: float = 1.0, condition_vector: "SteeringVector" = None, condition_layer_ids: List[int] = None, condition_vector_threshold: float = None, condition_comparator_threshold_is: str = "larger",  condition_threshold_comparison_mode: str = "mean",    use_explained_variance: bool = False, use_ooi_preventive_normalization: bool = False,  apply_behavior_on_first_call: bool = True, **kwargs) -> None:
        """
        Apply (conditional) activation steering to the model.

        This method configures the model to apply behavior modifications based on
        specified conditions (if given). It sets up both behavior and condition vectors across specified layers of the model.

        Args:
            behavior_vector (Optional[SteeringVector]): The vector representing the desired behavior change.
            behavior_layer_ids (List[int]): Layers to apply the behavior vector to. Default is [10, 11, 12, 13, 14, 15].
            behavior_vector_strength (float): Scaling factor for the behavior vector. Default is 1.0.
            condition_vector (SteeringVector): The vector representing the condition for applying the behavior.
            condition_layer_ids (List[int]): Layers to check the condition on.
            condition_vector_threshold (float): Threshold for condition activation.
            condition_comparator_threshold_is (str): Whether to activate when similarity is "larger" or "smaller" than threshold. Default is "larger".
            condition_threshold_comparison_mode (str): How to compare thresholds, either "mean" or "last". Default is "mean".
            use_explained_variance (bool): Whether to scale vectors by their explained variance. Default is False.
            use_ooi_preventive_normalization (bool): Whether to use out-of-input preventive normalization. Default is False.
            apply_behavior_on_first_call (bool): Whether to apply behavior vector on the first forward call. Default is True.
            **kwargs: Additional keyword arguments to pass to the LeashLayer's steer method.

        Raises:
            ValueError: If only one of condition_layer_ids or condition_vector is given. Omitting both is okay.

        Note:
            - This method updates both class and instance attributes of LeashLayer.
            - The behavior vector is applied only if a condition vector is not specified or if the condition is met.
            - Condition checking occurs only in specified layers, while behavior modification can be applied to different layers.

        """
        log(f"Steering...", style="bold", class_name="MalleableModel")

        layers = get_model_layer_list(self.model)
        num_layers = len(layers)

        if (condition_layer_ids is None) != (condition_vector is None):
            raise ValueError("condition_layer_ids and condition_vector must be both given or both not given")

        # Create boolean lists for condition and behavior layers
        condition_layers = [False] * num_layers
        behavior_layers = [False] * num_layers
    
        if condition_layer_ids:
            for layer_id in condition_layer_ids:
                condition_layers[layer_id] = True
        
        if behavior_vector is not None:
            #log(f"Applying behavior steering to layers: {behavior_layer_ids}", class_name="MalleableModel")
            for layer_id in behavior_layer_ids:
                behavior_layers[layer_id] = True

        # Update LeashLayer class attributes
        LeashLayer.condition_layers = {i: v for i, v in enumerate(condition_layers)}
        LeashLayer.behavior_layers = {i: v for i, v in enumerate(behavior_layers)}

        # Update LeashLayer instance attributes
        for layer_id in range(len(layers)):
            layer = layers[layer_id]
            behavior_tensor = None
            if behavior_vector is not None:
                if layer_id in behavior_layer_ids:
                    if use_explained_variance:
                        behavior_direction = use_explained_variance(behavior_vector)
                    else:
                        behavior_direction = behavior_vector.directions[layer_id]

                    behavior_tensor = torch.tensor(behavior_vector_strength * behavior_direction, dtype=self.model.dtype).to(self.model.device)

            
            condition_projector = None
            if condition_vector is not None and layer_id in condition_layer_ids:
                condition_direction = condition_vector.directions[layer_id]
                if use_explained_variance:
                    condition_direction = use_explained_variance(condition_vector)
                else:
                    condition_direction = condition_vector.directions[layer_id]
                
                condition_tensor = torch.tensor(condition_direction, dtype=self.model.dtype).to(self.model.device)
                condition_projector = torch.ger(condition_tensor, condition_tensor) / torch.dot(condition_tensor, condition_tensor)
            
            layer.steer(
                behavior_vector=behavior_tensor, 
                condition_projector=condition_projector, 
                threshold=condition_vector_threshold, 
                use_ooi_preventive_normalization=use_ooi_preventive_normalization,
                apply_behavior_on_first_call=apply_behavior_on_first_call,
                condition_comparator_threshold_is=condition_comparator_threshold_is,
                condition_threshold_comparison_mode=condition_threshold_comparison_mode,
                **kwargs
            )

    def multisteer(self, behavior_vectors: List[Optional["SteeringVector"]], behavior_layer_ids: List[List[int]], behavior_vector_strengths: List[float], condition_vectors: List["SteeringVector"], condition_layer_ids: List[List[int]], condition_vector_thresholds: List[float], condition_comparator_threshold_is: List[str], rules: List[str], condition_threshold_comparison_modes: List[str] = None, use_explained_variance: bool = False, use_ooi_preventive_normalization: bool = False, apply_behavior_on_first_call: bool = True, **kwargs) -> None:
        """
        Apply multiple conditional steering rules to the model.

        This method configures the model to apply multiple behavior modifications
        based on multiple specified conditions. It allows for complex steering
        scenarios with different behaviors triggered by different conditions.

        Args:
            behavior_vectors (List[Optional[SteeringVector]]): List of vectors representing desired behavior changes.
            behavior_layer_ids (List[List[int]]): List of layers to apply each behavior vector to.
            behavior_vector_strengths (List[float]): List of scaling factors for each behavior vector.
            condition_vectors (List[SteeringVector]): List of vectors representing conditions for applying behaviors.
            condition_layer_ids (List[List[int]]): List of layers to check each condition on.
            condition_vector_thresholds (List[float]): List of thresholds for condition activations.
            condition_comparator_threshold_is (List[str]): List specifying whether to activate when similarity is "larger" or "smaller" than threshold for each condition.
            rules (List[str]): List of rules specifying how conditions trigger behaviors (e.g., "if C1 then B1", "if C2 or C3 then B2").
            condition_threshold_comparison_modes (List[str]): List specifying how to compare thresholds for each condition, either "mean" or "last". Default is ["mean"] * num_conditions if None.
            use_explained_variance (bool): Whether to scale vectors by their explained variance. Default is False.
            use_ooi_preventive_normalization (bool): Whether to use out-of-input preventive normalization. Default is False.
            apply_behavior_on_first_call (bool): Whether to apply behavior vectors on the first forward call. Default is True.
            **kwargs: Additional keyword arguments to pass to the LeashLayer's multisteer method.

        Raises:
            AssertionError: If there's a mismatch in the lengths of condition or behavior parameter lists.

        Note:
            - This method allows for complex steering scenarios with multiple conditions and behaviors.
            - Each condition can be checked on different layers, and each behavior can be applied to different layers.
            - The rules parameter allows for logical combinations of conditions to trigger specific behaviors.
            - Ensure that the lengths of all list parameters match the number of conditions or behaviors as appropriate.
        """
        log(f"Multi-steering...", style="bold", class_name="MalleableModel")

        layers = get_model_layer_list(self.model)
        num_layers = len(layers)
        num_conditions = len(condition_vectors)
        num_behaviors = len(behavior_vectors)

        if condition_threshold_comparison_modes is None:
            condition_threshold_comparison_modes = ["mean"] * num_conditions
        # Validate input lengths
        assert len(condition_vectors) == len(condition_layer_ids) == len(condition_comparator_threshold_is) == len(condition_vector_thresholds) == len(condition_threshold_comparison_modes), "Mismatch in condition parameters"
        assert len(behavior_vectors) == len(behavior_layer_ids) == len(behavior_vector_strengths), "Mismatch in behavior parameters"

        # Create separate boolean lists for each condition and behavior
        condition_layers = [{i: False for i in range(num_layers)} for _ in range(num_conditions)]
        behavior_layers = [{i: False for i in range(num_layers)} for _ in range(num_behaviors)]

        for i, condition_layers_ids in enumerate(condition_layer_ids):
            for layer_id in condition_layers_ids:
                condition_layers[i][layer_id] = True

        for i, behavior_layers_ids in enumerate(behavior_layer_ids):
            for layer_id in behavior_layers_ids:
                behavior_layers[i][layer_id] = True

        # Update LeashLayer class attributes
        LeashLayer.condition_layers = condition_layers
        LeashLayer.behavior_layers = behavior_layers

        # Update LeashLayer instance attributes
        for layer_id in range(num_layers):
            layer = layers[layer_id]
            behavior_tensors = []
            condition_projectors = []

            for i in range(num_conditions):
                condition_projector = None
                if layer_id in condition_layer_ids[i]:
                    condition_direction = condition_vectors[i].directions[layer_id]
                    if use_explained_variance:
                        condition_direction = self.use_explained_variance(condition_vectors[i])
                    condition_tensor = torch.tensor(condition_direction, dtype=self.model.dtype).to(self.model.device)
                    condition_projector = torch.ger(condition_tensor, condition_tensor) / torch.dot(condition_tensor, condition_tensor)
                condition_projectors.append(condition_projector)

            for i in range(num_behaviors):
                behavior_tensor = None
                if behavior_vectors[i] is not None and layer_id in behavior_layer_ids[i]:
                    behavior_direction = behavior_vectors[i].directions[layer_id]
                    if use_explained_variance:
                        behavior_direction = self.use_explained_variance(behavior_vectors[i])
                    behavior_tensor = torch.tensor(behavior_vector_strengths[i] * behavior_direction, dtype=self.model.dtype).to(self.model.device)
                behavior_tensors.append(behavior_tensor)
 
            layer.multisteer(
                behavior_vectors=behavior_tensors,
                condition_projectors=condition_projectors,
                thresholds=condition_vector_thresholds,
                use_ooi_preventive_normalization=use_ooi_preventive_normalization,
                apply_behavior_on_first_call=apply_behavior_on_first_call,
                condition_comparator_threshold_is=condition_comparator_threshold_is,
                condition_threshold_comparison_modes=condition_threshold_comparison_modes,
                rules=rules,
                **kwargs
            )

        log(f"Multi-steering set up with {num_conditions} conditions and {num_behaviors} behaviors", class_name="MalleableModel")


    def reset_leash_to_default(self) -> None:
        """
        Reset the model's steering configuration to its default state.

        This method removes all applied steering configurations, including
        behavior vectors and condition vectors, from all layers of the model.
        It resets both instance-specific and class-wide attributes of the
        LeashLayer wrapper.

        Returns:
            None

        Note:
            - This method should be called when you want to clear all steering
            configurations and return the model to its original behavior.
            - It's useful when you want to apply a new steering configuration
            from scratch or when you're done with steering and want to use
            the model in its default state.
            - This reset affects all layers of the model simultaneously.
        """
        log("Resetting leash to default...", style="bold", class_name="MalleableModel")
        layers = get_model_layer_list(self.model)
        for layer in layers:
            layer.reset_instance()
        LeashLayer.reset_class()


    def generate(self, *args, **kwargs):
        """
        Generate output using the underlying model.

        This method is a pass-through to the generate method of the wrapped model.
        It allows for text generation using the model's native generation capabilities,
        which may include techniques like beam search, sampling, or others depending
        on the underlying model architecture.

        Args:
            *args: Positional arguments to pass to the underlying model's generate method.
            **kwargs: Keyword arguments to pass to the underlying model's generate method.

        Returns:
            The output generated by the underlying model's generate method. The exact
            return type depends on the specific model and the provided arguments.

        Note:
            - The behavior of this method is determined by the underlying model and
            the arguments passed to it.
            - Any steering configurations applied to the model will affect the 
            generation process.
            - For detailed information on available arguments and their effects,
            refer to the documentation of the specific pre-trained model being used.
        """
        return self.model.generate(*args, **kwargs)


    def respond(self, prompt, settings=None, use_chat_template=True,reset_after_response=True):
        """
        Generate a response to a given prompt using the underlying language model.

        Args:
            prompt: The input prompt to generate a response for.
            settings: A dictionary of generation settings. If None, default settings are used.
            use_chat_template: Whether to apply the chat template to the prompt.
            reset_after_response: Whether to reset the model's internal state after generating a response.

        Returns:
            The generated response text.
        """
        # Force model to CPU or GPU to ensure weights are in the correct device
        self.model.to(self.device)
        
        if use_chat_template:
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{prompt}"}],
                tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        if settings is None:
            settings = {
                "pad_token_id": self.tokenizer.eos_token_id,
                "do_sample": False,
                "max_new_tokens": 50,
                "repetition_penalty": 1.1,
            }
        
        with torch.no_grad():  # Ensure we're not tracking gradients during inference
            output = self.model.generate(**input_ids, **settings)
        
        response = self.tokenizer.decode(output.squeeze()[input_ids['input_ids'].shape[1]:])

        if reset_after_response:
            # reset for each call
            LeashLayer.condition_met = defaultdict(lambda: False)
            LeashLayer.forward_calls = defaultdict(int)
            LeashLayer.condition_similarities = defaultdict(lambda: defaultdict(float))

        return response


    def respond_batch_sequential(self, prompts, settings=None, use_chat_template=True):
        self.model.to(self.device)
        """
        Generate responses for multiple prompts sequentially.

        Args:
            prompts: A list of input prompts to generate responses for.
            settings: A dictionary of generation settings. If None, default settings are used.
            use_chat_template: Whether to apply the chat template to each prompt.

        Returns:
            A list of generated response texts, one for each input prompt.
        """
        responses = []
        for prompt in prompts:
            response = self.respond(prompt, settings, use_chat_template)
            responses.append(response)

        return responses
    

    def find_best_condition_point(self, positive_strings: List[str], negative_strings: List[str], condition_vector: 'SteeringVector', layer_range: Optional[Tuple[int, int]] = None, max_layers_to_combine: int = 1, threshold_range: Tuple[float, float] = (0.0, 1.0), threshold_step: float = 0.01, save_analysis: bool = False, file_path: Optional[str] = None, condition_threshold_comparison_mode: str = "mean") -> Tuple[List[int], float, str, float]:
        """
        Find the optimal condition point for steering.

        Args:
            positive_strings: List of strings that should trigger the condition.
            negative_strings: List of strings that should not trigger the condition.
            condition_vector: The steering vector representing the condition.
            layer_range: Range of layers to search for the condition point.
            max_layers_to_combine: Maximum number of layers to combine in the search.
            threshold_range: Range of thresholds to search.
            threshold_step: Step size for threshold search.
            save_analysis: Whether to save the analysis results.
            file_path: Path to save the analysis results.
            condition_threshold_comparison_mode: Mode for comparing condition thresholds.

        Returns:
            A tuple containing the best layers, threshold, direction, and F1 score.
        """
        if layer_range is None:
            layer_range = (1, len(get_model_layer_list(self.model)))

        log(f"Initializing search for best condition point...", style="bold", class_name="MalleableModel")

        all_strings = positive_strings + negative_strings
        y_true = [1] * len(positive_strings) + [0] * len(negative_strings)

        layers = list(range(*layer_range))
        best_f1 = 0
        best_config = None

        # Apply steering to all layers at once
        self.steer(
            condition_vector=condition_vector,
            condition_layer_ids=layers,
            condition_vector_threshold=1,  # Dummy threshold
            condition_comparator_threshold_is="smaller",  # Dummy direction
            apply_behavior_on_first_call=False,
            condition_threshold_comparison_mode=condition_threshold_comparison_mode
        )

        # Collect similarities for all strings and layers
        similarities = []
        for i, string in enumerate(custom_progress(all_strings, "Processing strings")):
            settings = {
                "pad_token_id": self.tokenizer.eos_token_id,
                "do_sample": False,
                "max_new_tokens": 1,
                "repetition_penalty": 1.1,
            }
            self.respond(string, settings = settings, reset_after_response = False)
            similarities.append({layer: LeashLayer.condition_similarities[0][layer] for layer in layers})
            LeashLayer.condition_met = defaultdict(lambda: False)
            LeashLayer.forward_calls = defaultdict(int)
            LeashLayer.condition_similarities = defaultdict(lambda: defaultdict(float))

        # Create a list of all combinations to iterate over
        all_combinations = [
            (r, layer_combo, threshold, direction)
            for r in range(1, min(max_layers_to_combine, len(layers)) + 1)
            for layer_combo in combinations(layers, r)
            for threshold in np.arange(*threshold_range, threshold_step)
            for direction in ['larger', 'smaller']
        ]

        # Find best combination
        analysis_results = {}
        for r, layer_combo, threshold, direction in custom_progress(all_combinations, "Searching for best condition point"):
            layer_key = f"layers_{'_'.join(map(str, layer_combo))}"
            if layer_key not in analysis_results:
                analysis_results[layer_key] = {"f1_scores": {}, "similarities": {}}

            y_pred = []
            for i, sim_dict in enumerate(similarities):
                condition_met = any(
                    (sim_dict[layer] > threshold) == (direction == 'smaller')
                    for layer in layer_combo
                )
                y_pred.append(1 if condition_met else 0)

            f1 = f1_score(y_true, y_pred)
            if f1 > 0:  # Only record non-zero F1 scores
                analysis_results[layer_key]["f1_scores"][f"{threshold:.3f}_{direction}"] = f1

            if f1 > best_f1:
                best_f1 = f1
                best_config = (list(layer_combo), threshold, direction)

        # Record similarities per layer
        for layer in layers:
            analysis_results[f"layer_{layer}"] = {
                "similarities": {
                    "positive": [sim_dict[layer] for sim_dict in similarities[:len(positive_strings)]],
                    "negative": [sim_dict[layer] for sim_dict in similarities[len(positive_strings):]]
                }
            }

        log(f"Search completed.", style="bold", class_name="MalleableModel")
        rounded_threshold = round(best_config[1], 3)
        log(f"Best condition point found: Layers {best_config[0]}, Threshold {rounded_threshold:.3f}, Direction '{best_config[2]}', F1 Score {best_f1:.3f}", style="bold green", class_name="MalleableModel")

        if save_analysis:
            self._save_analysis_results(analysis_results, best_config[0], rounded_threshold, best_config[2], best_f1, file_path)

        self.reset_leash_to_default()
        return best_config[0], rounded_threshold, best_config[2], best_f1


    def _save_analysis_results(self, analysis_results, best_layers, best_threshold, best_direction, best_f1, file_path):
        """
        Save the analysis results from find_best_condition_point to a file.

        Args:
            analysis_results: Dictionary containing the analysis results.
            best_layers: List of layers that gave the best performance.
            best_threshold: The threshold value that gave the best performance.
            best_direction: The direction ('larger' or 'smaller') that gave the best performance.
            best_f1: The best F1 score achieved.
            file_path: Path to save the analysis results.
        """
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # If no file name is provided, generate a default one
        if not os.path.basename(file_path):
            file_name = f"condition_point_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(directory, file_name)

        summary = {
            "best_layers": best_layers,
            "best_threshold": best_threshold,
            "best_direction": best_direction,
            "best_f1_score": best_f1,
            "analysis": analysis_results
        }

        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)

        log(f"Analysis results saved to {file_path}", style="bold blue", class_name="MalleableModel")


    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the model.

        This method delegates to the underlying model's forward method.

        Args:
            *args: Positional arguments to pass to the underlying model.
            **kwargs: Keyword arguments to pass to the underlying model.

        Returns:
            The output of the underlying model's forward pass.
        """
        return self.model(*args, **kwargs)


    def __call__(self, *args, **kwargs):
        """
        Make the MalleableModel instance callable.

        This method allows the MalleableModel to be used like a function, delegating to the underlying model.

        Args:
            *args: Positional arguments to pass to the underlying model.
            **kwargs: Keyword arguments to pass to the underlying model.

        Returns:
            The output of the underlying model.
        """
        return self.model(*args, **kwargs)


def get_model_layer_list(model: MalleableModel | PreTrainedModel) -> torch.nn.ModuleList:
    """
    Get the list of layers from a model.

    This function handles different model architectures to retrieve their layers.

    Args:
        model: Either a MalleableModel or a PreTrainedModel.

    Returns:
        A ModuleList containing the model's layers.

    Raises:
        ValueError: If the function doesn't know how to get layers for the given model type.
    """
    if isinstance(model, MalleableModel):
        model = model.model  # Use the underlying model if the model is a MalleableModel instance

    if hasattr(model, "model"):  # mistral-like
        return model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        return model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")
