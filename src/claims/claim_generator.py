import importlib
import json
from typing import List, Type

import requests
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

    @staticmethod
    def build_claim_extraction_prompt(text: str) -> str:
        return REFINED_CLAIM_PROMPT.format(SOURCE_TEXT=text)


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


class APIClaimGenerator(BaseClaimGenerator):
    """
    Generic API-based claim generator that can talk to either OpenAI
    or a local Jan-based endpoint, depending on config.type.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = config.device
        # No local model, so skip:
        self.model = None
        self.tokenizer = None

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        predictions = []
        for batch in tqdm(self._chunked(texts, self.config.batch_size),
                          desc=f"Generating claims with {self.config.model_name} [API]"):
            # Build the prompts or messages
            prompts_text = [self.build_claim_extraction_prompt(t) for t in batch]

            # If using an OpenAI/Jan "chat" format, build messages:
            # Typically you might do one request per text for simplicity:
            for prompt in prompts_text:
                # We'll do a single user message
                messages = [{"role": "user", "content": prompt}]
                response_json = self._send_api_request(messages)
                claims_for_this_text = self._parse_api_response(response_json)
                predictions.append(claims_for_this_text)

        return predictions

    def _build_prompt(self, text: str) -> dict:
        """
        For an OpenAI/Jan chat endpoint, you typically pass a "messages" list.
        Adjust as needed for your local Jan or OpenAI format.
        """
        return {
            "role": "user",
            "content": f"Extract atomic factual claims from this text:\n\n{text}\n\nReturn JSON with a 'claims' key."
        }

    def _send_api_request(self, prompts: List[dict]) -> dict:

        YOUR_OPENAI_API_KEY = "placeholder"

        """
        Actually sends a request to either the local Jan server or the OpenAI endpoint,
        depending on config.type. We'll do a single request with multiple prompt messages
        if your endpoint supports it. Otherwise, loop prompts one-by-one.
        """
        # Distinguish endpoints
        api_url = self.config.model_name  # e.g., "http://127.0.0.1:1337/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        if self.config.type == "openai":
            # Real OpenAI
            headers["Authorization"] = f"Bearer {YOUR_OPENAI_API_KEY}"
        else:
            # Possibly no auth needed for local Jan, or some local token
            pass

        # For OpenAI/Jan, the JSON structure is typically:
        payload = {
            "model": "llama3.2-1b-instruct",  # or any local model alias
            "messages": prompts,
            "temperature": 0.0
        }

        # If you need multiple completions in one request, you can do that,
        # or else do a single prompt at a time. For debugging, single prompt is simpler.

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def _parse_api_response(self, response_data: dict) -> List[List[str]]:
        """
        Interpret the response. For an OpenAI/Jan chat completion:
          response_data["choices"][0]["message"]["content"]
        should contain the text we need. Then parse JSON from that text.
        """
        all_claims = []
        choices = response_data.get("choices", [])
        for choice in choices:
            content = choice["message"]["content"]
            try:
                data = json.loads(content)
                claims = data.get("claims", [])
                all_claims.append(claims if isinstance(claims, list) else [])
            except json.JSONDecodeError:
                all_claims.append([])
        return all_claims
