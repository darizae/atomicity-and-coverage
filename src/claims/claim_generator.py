import importlib
import json
from typing import List, Type
from tqdm import tqdm

from src.config.prompt_templates import REFINED_CLAIM_PROMPT


class ModelConfig:
    def __init__(self, model_name: str, tokenizer_class_path: str, model_class_path: str,
                 device: str = "cpu", batch_size: int = 32, max_length: int = 512, truncation: bool = True):
        self.model_name = model_name
        self.tokenizer_class_path = tokenizer_class_path
        self.model_class_path = model_class_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.truncation = truncation


class BaseClaimGenerator:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        self.model, self.tokenizer = self.load_model_and_tokenizer(config)

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        raise NotImplementedError("Subclasses must implement generate_claims")

    def load_model_and_tokenizer(self, config: ModelConfig):
        """Dynamically loads and initializes the model and tokenizer."""
        tokenizer_cls = self._import_class(config.tokenizer_class_path)
        model_cls = self._import_class(config.model_class_path)

        tokenizer = tokenizer_cls.from_pretrained(config.model_name)
        model = model_cls.from_pretrained(config.model_name).to(config.device)
        return model, tokenizer

    def _generate(self, inputs):
        """Handles text generation and decoding."""
        outputs = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @staticmethod
    def _import_class(class_path: str) -> Type:
        module_name, cls_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, cls_name)

    @staticmethod
    def _chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]


class Seq2SeqClaimGenerator(BaseClaimGenerator):
    """Claim generator using sequence-to-sequence models (T5, BART, etc.)."""

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        predictions = []
        for batch in tqdm(self._chunked(texts, self.config.batch_size),
                          desc=f"Generating claims with {self.config.model_name} [Seq2Seq]"):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=self.config.truncation,
                                    max_length=self.config.max_length).to(self.device)
            decoded = self._generate(inputs)
            print(f"Decoded outputs: {decoded}")
            predictions.extend([text.split(". ") for text in decoded])
        return predictions


class CausalLMClaimGenerator(BaseClaimGenerator):
    """Claim generator using autoregressive causal language models."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.pad_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        predictions = []
        for batch in tqdm(self._chunked(texts, self.config.batch_size),
                          desc=f"Generating claims with {self.config.model_name} [CausalLM]"):
            prompts = [self.build_claim_extraction_prompt(t) for t in batch]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            decoded = self._generate(inputs)
            predictions.extend(self._parse_json_outputs(decoded))
        return predictions

    @staticmethod
    def build_claim_extraction_prompt(text: str) -> str:
        return REFINED_CLAIM_PROMPT.format(SOURCE_TEXT=text)

    @staticmethod
    def _parse_json_outputs(decoded_outputs: List[str]) -> List[List[str]]:
        print(f"Decoded outputs: {decoded_outputs}")
        parsed_claims = []
        for output in decoded_outputs:
            try:
                data = json.loads(output.strip())
                claims = data.get("claims", [])
                parsed_claims.append(claims if isinstance(claims, list) else [])
            except json.JSONDecodeError:
                parsed_claims.append([])
        return parsed_claims


class OpenAIClaimGenerator(BaseClaimGenerator):
    """Placeholder class for an OpenAI API-based claim generator."""

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        # This will use OpenAI's API to generate claims, needs implementation
        raise NotImplementedError("OpenAIClaimGenerator is not yet implemented")
