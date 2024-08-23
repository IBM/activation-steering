from typing import List, Literal, Tuple, Optional
from activation_steering.utils import ContrastivePair
from transformers import PreTrainedTokenizerBase

from activation_steering.utils import return_default_suffixes
from activation_steering.config import log, GlobalConfig

class SteeringDataset:
    """
    Create a formatted dataset for steering a language model.

    This class takes a list of examples (either contrastive messages or contrastive text) 
    and a tokenizer, and formats the examples into a dataset of ContrastivePair objects.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        examples: List,
        suffixes: List[Tuple[str, str]] = None,
        disable_suffixes: bool = False,
        use_chat_template: bool = True,
        system_message: Optional[Tuple[str, str]] = None
    ):
        """
        Initialize the SteeringDataset.

        Args:
            tokenizer: The tokenizer used to tokenize and format the examples.
            examples: A list of examples, either contrastive messages or contrastive text.
            suffixes: A list of suffixes to append to the formatted dataset. If None, default suffixes will be used.
            disable_suffixes: If True, no suffixes will be appended to the examples.
            use_chat_template: If True, applies the chat template to the examples.
            system_message: Optional system message to be included in the chat template.
        """
        self.tokenizer = tokenizer
        self.suffixes = suffixes
        self.formatted_dataset = []
        self.formatted_dataset_pre_populated = []
        self.use_chat_template = use_chat_template

        log(f"Processing {len(examples)} examples", class_name="SteeringDataset")

        for example in examples:
            if self.use_chat_template:
                if system_message:
                    message_a = [{"role": "system", "content": f"{system_message[0]}"}, {"role": "user", "content": f"{self.clean_text(example[0])}"}]
                    message_b = [{"role": "system", "content": f"{system_message[1]}"}, {"role": "user", "content": f"{self.clean_text(example[1])}"}]
                else:
                    message_a = [{"role": "user", "content": f"{self.clean_text(example[0])}"}]
                    message_b = [{"role": "user", "content": f"{self.clean_text(example[1])}"}]
                positive = tokenizer.apply_chat_template(message_a, tokenize=False, add_generation_prompt=False)
                negative = tokenizer.apply_chat_template(message_b, tokenize=False, add_generation_prompt=False)
            else:
                positive = self.clean_text(example[0])
                negative = self.clean_text(example[1])
            
            self.formatted_dataset_pre_populated.append(
                ContrastivePair(positive=positive, negative=negative)
            )

        log(f"Processed {len(self.formatted_dataset_pre_populated)} examples", class_name="SteeringDataset")

        # Handle suffixes (same as original)
        if suffixes is not None and not disable_suffixes and isinstance(suffixes[0], tuple):
            for positive_suffix, negative_suffix in suffixes:
                for pair in self.formatted_dataset_pre_populated:
                    self.formatted_dataset.append(
                        ContrastivePair(
                            positive=pair.positive + positive_suffix,
                            negative=pair.negative + negative_suffix
                        )
                    )
        elif suffixes is not None and not disable_suffixes and isinstance(suffixes[0], str):
            for suffix in suffixes:
                for pair in self.formatted_dataset_pre_populated:
                    self.formatted_dataset.append(
                        ContrastivePair(
                            positive=pair.positive + suffix,
                            negative=pair.negative + suffix
                        )
                    )
        elif suffixes is None and not disable_suffixes:
            default_suffixes = return_default_suffixes()
            for suffix in default_suffixes:
                for pair in self.formatted_dataset_pre_populated:
                    self.formatted_dataset.append(
                        ContrastivePair(
                            positive=pair.positive + suffix,
                            negative=pair.negative + suffix
                        )
                    )
        else:
            self.formatted_dataset = self.formatted_dataset_pre_populated
        
        log("=*"*15)
        log(f"[bold green]Final dataset size[/bold green]: {len(self.formatted_dataset)} examples", class_name="SteeringDataset")
        log(f"[bold red]Positive example[/bold red]: {self.formatted_dataset[0].positive}", class_name="SteeringDataset")
        log(f"[bold blue]Negative example[/bold blue]: {self.formatted_dataset[0].negative}", class_name="SteeringDataset")
        log("=*"*15)
        

    def clean_text(self, text: str) -> str:
        """
        Clean the input text by replacing special tokens.

        Args:
            text: The input text to be cleaned.

        Returns:
            The cleaned text with special tokens replaced.
        """
        if not text:
            return text

        def insert_vline(token: str) -> str:
            if len(token) < 2:
                return " "
            elif len(token) == 2:
                return f"{token[0]}|{token[1]}"
            else:
                return f"{token[:1]}|{token[1:-1]}|{token[-1:]}"

        if self.tokenizer.bos_token:
            text = text.replace(self.tokenizer.bos_token, insert_vline(self.tokenizer.bos_token))
        if self.tokenizer.eos_token:
            text = text.replace(self.tokenizer.eos_token, insert_vline(self.tokenizer.eos_token))
        if self.tokenizer.pad_token:
            text = text.replace(self.tokenizer.pad_token, insert_vline(self.tokenizer.pad_token))
        if self.tokenizer.unk_token:
            text = text.replace(self.tokenizer.unk_token, insert_vline(self.tokenizer.unk_token))

        return text