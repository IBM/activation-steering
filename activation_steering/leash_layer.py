import torch
import torch.nn as nn
from typing import List
from collections import defaultdict
from activation_steering.utils import LayerControlParams
from activation_steering.config import log

class LeashLayer(nn.Module):
    """
    A wrapper layer that implements conditional activation steering for language models.

    This layer can be applied to existing model layers to enable fine-grained control
    over the model's behavior through steering and conditional activation.

    Class Attributes:
        condition_met: A defaultdict tracking whether conditions have been met.
        forward_calls: A defaultdict counting forward passes for each layer.
        condition_layers: Tracks which layers are condition layers.
        behavior_layers: Tracks which layers are behavior layers.
        condition_similarities: Stores condition similarities for each layer.
    """
    condition_met = defaultdict(bool)
    forward_calls = defaultdict(int)
    condition_layers = None
    behavior_layers = None
    condition_similarities = defaultdict(lambda: defaultdict(float)) # this is used later to find the best condition point

    def __init__(self, layer: nn.Module, layer_id: int) -> None:
        """
        Initialize a LeashLayer.

        Args:
            layer: The underlying layer to be wrapped.
            layer_id: The ID of this layer in the model.
        """
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.use_ooi_preventive_normalization = False
        self.apply_behavior_on_first_call = True
        self.is_multi_steering = False
        #self.reset_instance()

    def steer(self, behavior_vector: torch.Tensor, condition_projector: torch.Tensor, threshold: float = 0.0, use_ooi_preventive_normalization: bool = True, apply_behavior_on_first_call: bool = True, condition_comparator_threshold_is: str = "larger", condition_threshold_comparison_mode: str = "mean", **kwargs) -> None:
        """
        Configure steering for this layer.

        Args:
            behavior_vector: The behavior vector to apply.
            condition_projector: The condition projector to use.
            threshold: The threshold for condition activation.
            use_ooi_preventive_normalization: Whether to use OOI preventive normalization.
            apply_behavior_on_first_call: Whether to apply behavior on the first forward call.
            condition_comparator_threshold_is: How to compare the condition to the threshold.
            condition_threshold_comparison_mode: How to compute the condition value.
            **kwargs: Additional parameters for LayerControlParams.
        """
        self.is_multi_steering = False
        self.behavior_vector = behavior_vector
        self.condition_projector = condition_projector
        self.threshold = threshold
        self.params = LayerControlParams(**kwargs)
        self.use_ooi_preventive_normalization = use_ooi_preventive_normalization
        self.apply_behavior_on_first_call = apply_behavior_on_first_call
        self.condition_comparator_threshold_is = condition_comparator_threshold_is
        self.condition_threshold_comparison_mode = condition_threshold_comparison_mode
        log(f"    Steering set with apply_behavior_on_first_call: {self.apply_behavior_on_first_call}", class_name="LeashLayer")

    def multisteer(self, behavior_vectors: List[torch.Tensor], condition_projectors: List[torch.Tensor], thresholds: List[float], use_ooi_preventive_normalization: bool = True, apply_behavior_on_first_call: bool = True, condition_comparator_threshold_is: List[str] = ["larger"], condition_threshold_comparison_modes: List[str] = ["mean"], rules: List[str] = None, **kwargs) -> None:
        """
        Configure multi-steering for this layer.

        Args:
            behavior_vectors: List of behavior vectors to apply.
            condition_projectors: List of condition projectors to use.
            thresholds: List of thresholds for condition activation.
            use_ooi_preventive_normalization: Whether to use OOI preventive normalization.
            apply_behavior_on_first_call: Whether to apply behavior on the first forward call.
            condition_comparator_threshold_is: How to compare each condition to its threshold.
            condition_threshold_comparison_modes: How to compute each condition value.
            rules: List of rules for applying behaviors based on conditions.
            **kwargs: Additional parameters for LayerControlParams.
        """
        self.is_multi_steering = True
        self.behavior_vectors = behavior_vectors
        self.condition_projectors = condition_projectors
        self.thresholds = thresholds
        self.params = [LayerControlParams(**kwargs) for _ in range(len(behavior_vectors))]
        self.use_ooi_preventive_normalization = use_ooi_preventive_normalization
        self.apply_behavior_on_first_call = apply_behavior_on_first_call
        self.condition_comparator_threshold_is = condition_comparator_threshold_is
        self.condition_threshold_comparison_modes = condition_threshold_comparison_modes
        self.rules = rules
        log(f"    Multi-steering set for {len(condition_projectors)} conditions and {len(behavior_vectors)} behaviors", class_name="LeashLayer")

    def forward(self, hidden_states, *args, **kwargs):
        """
        Perform a forward pass through this layer, applying steering if configured.

        Args:
            hidden_states: The input hidden states.
            *args: Additional positional arguments for the underlying layer.
            **kwargs: Additional keyword arguments for the underlying layer.

        Returns:
            The output of the underlying layer, potentially modified by steering.
        """
        LeashLayer.forward_calls[self.layer_id] += 1
        batch_size, seq_length, hidden_dim = hidden_states.shape
        log(f"\n\nThis is forward_call {LeashLayer.forward_calls[self.layer_id]} @ Layer {self.layer_id}", class_name="LeashLayer")
        log(f"    Sequence length is {seq_length}", class_name="LeashLayer")

        if not self.is_multi_steering:
            # is a dict
            if LeashLayer.condition_layers == None:
                # CASE 1 -> no steering
                is_condition_layer = False
                is_behavior_layer = False
            else:
                # CASE 2 -> steering
                is_condition_layer = LeashLayer.condition_layers[self.layer_id]
                is_behavior_layer = LeashLayer.behavior_layers[self.layer_id]
        else:
            # is a list of dict
            # CASE 3 -> multi conditioned steering
            is_condition_layer = any(layers[self.layer_id] for layers in LeashLayer.condition_layers)
            is_behavior_layer = any(layers[self.layer_id] for layers in LeashLayer.behavior_layers)
        
        log(f"    is_condition_layer: {is_condition_layer}", class_name="LeashLayer")
        log(f"    is_behavior_layer: {is_behavior_layer}", class_name="LeashLayer")

        original_norm = hidden_states.norm(dim=-1, keepdim=True)

        if is_condition_layer:
            if not self.is_multi_steering:
                self._process_single_condition(hidden_states[0])
            else:
                self._process_multi_conditions(hidden_states[0])

        if is_behavior_layer:
            if not self.is_multi_steering:
                self._apply_single_behavior(hidden_states)
            else:
                self._apply_multi_behaviors(hidden_states)

        if self.use_ooi_preventive_normalization and is_behavior_layer:
            hidden_states = self._apply_ooi_normalization(hidden_states, original_norm)

        return self.layer(hidden_states, *args, **kwargs)

    def _process_single_condition(self, hidden_state):
        """
        Process a single condition for steering.

        Args:
            hidden_state: The hidden state to process.
        """
        if not LeashLayer.condition_met[0] and LeashLayer.forward_calls[self.layer_id] == 1:
            if self.condition_threshold_comparison_mode == "mean":
                hidden_state = hidden_state.mean(dim=0)
            elif self.condition_threshold_comparison_mode == "last":  
                hidden_state = hidden_state[-1, :]
            
            projected_hidden_state = torch.tanh(torch.matmul(self.condition_projector, hidden_state))
            condition_similarity = self.compute_similarity(hidden_state, projected_hidden_state).item()
            LeashLayer.condition_similarities[0][self.layer_id] = condition_similarity

            if self.condition_comparator_threshold_is == "smaller":
                condition_met = (condition_similarity > self.threshold)
            elif self.condition_comparator_threshold_is == "larger":
                condition_met = (condition_similarity < self.threshold)
            
            LeashLayer.condition_met[0] = condition_met
            
            log(f"    Similarity: {condition_similarity}", class_name="LeashLayer")
            log(f"    Threshold: {self.threshold}", class_name="LeashLayer")
            log(f"    Condition Met: {condition_met}", class_name="LeashLayer")

    def _process_multi_conditions(self, hidden_state):
        """
        Process multiple conditions for multi-steering.

        Args:
            hidden_state: The hidden state to process.
        """
        for condition_idx, condition_projector in enumerate(self.condition_projectors):
            if condition_projector is not None and \
                not LeashLayer.condition_met[condition_idx] and \
                LeashLayer.forward_calls[self.layer_id] == 1 and \
                LeashLayer.condition_layers[condition_idx][self.layer_id]:
                if self.condition_threshold_comparison_modes[condition_idx] == "mean":
                    hidden_state_for_condition = hidden_state.mean(dim=0)
                elif self.condition_threshold_comparison_modes[condition_idx] == "last":
                    hidden_state_for_condition = hidden_state[-1, :]
                
                projected_hidden_state = torch.tanh(torch.matmul(condition_projector, hidden_state_for_condition))
                condition_similarity = self.compute_similarity(hidden_state_for_condition, projected_hidden_state).item()

                if self.condition_comparator_threshold_is[condition_idx] == "smaller":
                    condition_met = (condition_similarity > self.thresholds[condition_idx])
                elif self.condition_comparator_threshold_is[condition_idx] == "larger":
                    condition_met = (condition_similarity < self.thresholds[condition_idx])
                
                LeashLayer.condition_met[condition_idx] = condition_met
                
                log(f"    Condition {condition_idx} - Similarity: {condition_similarity}", class_name="LeashLayer")
                log(f"    Condition {condition_idx} - Threshold: {self.thresholds[condition_idx]}", class_name="LeashLayer")
                log(f"    Condition {condition_idx} - Condition Met: {condition_met}", class_name="LeashLayer")

    def _apply_single_behavior(self, hidden_states):
        """
        Apply a single behavior vector to the hidden states.

        Args:
            hidden_states: The hidden states to modify.
        """
        should_apply = not any(LeashLayer.condition_layers.values()) or LeashLayer.condition_met[0]

        log(f"    Should Apply Behavior: {should_apply}", class_name="LeashLayer")

        if should_apply:
            control = self.behavior_vector.to(dtype=hidden_states.dtype)
            if LeashLayer.forward_calls[self.layer_id] == 1:
                if self.apply_behavior_on_first_call:
                    hidden_states[0] = self.params.operator(hidden_states[0], control)
                else:
                    log(f"    apply_behavior_on_first_call is False, skipping behavior vector application", class_name="LeashLayer")
            else:
                hidden_states[0] = self.params.operator(hidden_states[0], control)
                log(f"    Applying behavior vector to all tokens", class_name="LeashLayer")

    def _apply_multi_behaviors(self, hidden_states):
        """
        Apply multiple behavior vectors to the hidden states based on rules.

        Args:
            hidden_states: The hidden states to modify.
        """
        for rule in self.rules:
            behavior_index = int(rule.split('B')[1]) - 1
            if self._evaluate_rule(rule) and \
                LeashLayer.behavior_layers[behavior_index][self.layer_id]:
                #print(behavior_index)
                log(f"    Rule '{rule}' satisfied. Applying behavior {behavior_index}", class_name="LeashLayer")
                control = self.behavior_vectors[behavior_index].to(dtype=hidden_states.dtype)
                if LeashLayer.forward_calls[self.layer_id] == 1:
                    if self.apply_behavior_on_first_call:
                        hidden_states[0] = self.params[behavior_index].operator(hidden_states[0], control)
                    else:
                        log(f"    apply_behavior_on_first_call is False, skipping behavior vector application", class_name="LeashLayer")
                else:
                    hidden_states[0] = self.params[behavior_index].operator(hidden_states[0], control)
                    log(f"    Applying behavior vector to all tokens", class_name="LeashLayer")
            else:
                log(f"    Rule '{rule}' not satisfied.", class_name="LeashLayer")

    def _evaluate_rule(self, rule: str) -> bool:
        """
        Evaluate a steering rule.

        Args:
            rule: The rule to evaluate.

        Returns:
            Boolean indicating whether the rule is satisfied.
        """
        rule_parts = rule.split('then')
        if len(rule_parts) != 2:
            return False
        
        condition_part = rule_parts[0].strip().lower()
        conditions = condition_part.replace('if', '').strip().split()
        
        if 'or' in conditions:
            return any(self._check_single_condition(cond) for cond in conditions if cond not in ['or', 'and'])
        elif 'and' in conditions:
            return all(self._check_single_condition(cond) for cond in conditions if cond not in ['or', 'and'])
        else:
            return self._check_single_condition(conditions[0])

    def _check_single_condition(self, condition: str) -> bool:
        """
        Check if a single condition is met.

        Args:
            condition: The condition to check.

        Returns:
            Boolean indicating whether the condition is met.
        """
        if condition.startswith('c'):
            try:
                condition_index = int(condition[1:]) - 1
                return LeashLayer.condition_met[condition_index]
            except (ValueError, IndexError):
                return False
        return False

    def compute_similarity(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute the cosine similarity between two tensors.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            The cosine similarity as a float.
        """
        return torch.dot(x.flatten(), y.flatten()) / (torch.norm(x) * torch.norm(y))

    def _apply_ooi_normalization(self, hidden_states, original_norm):
        """
        Apply out-of-input (OOI) preventive normalization to hidden states.

        Args:
            hidden_states: The hidden states to normalize.
            original_norm: The original norm of the hidden states.

        Returns:
            The normalized hidden states.
        """
        new_norm = hidden_states.norm(dim=-1, keepdim=True)
        max_ratio = (new_norm / original_norm).max().item()
        has_nan_inf = torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any()

        if max_ratio > 1 or has_nan_inf:
            log(f"    Applying OOI preventive normalization. Max_ratio was {max_ratio}", class_name="LeashLayer")
            hidden_states = hidden_states * (original_norm / new_norm)
        else:
            log(f"    No OOI preventive normalization. Max_ratio was {max_ratio}", class_name="LeashLayer")

        return hidden_states

    def reset_instance(self) -> None:
        """
        Reset this instance of LeashLayer to its default state.
        """
        log(f"    Resetting LeashLayer @ {self.layer_id} Instance Attributes", class_name="LeashLayer")
        self.params = LayerControlParams.default()
        self.condition_projector = None
        self.behavior_vector = None
        self.threshold = 0.0

    @classmethod
    def reset_class(cls) -> None:
        """
        Reset the class-level attributes of LeashLayer.
        """
        log(f"    Resetting LeashLayer Class Attributes", class_name="LeashLayer")
        cls.condition_met.clear()
        cls.forward_calls.clear()
        cls.condition_layers = None
        cls.behavior_layers = None
        cls.condition_similarities = defaultdict(lambda: defaultdict(float))