import importlib
import json
import os
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Type, Optional

import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import openai

from src.claims.prompt_templates import REFINED_CLAIM_PROMPT


@dataclass
class ModelConfig:
    model_name_or_path: str  # HF model name or path, OR engine name for Jan/OpenAI

    # For HF:
    tokenizer_class: Optional[str] = None
    model_class: Optional[str] = None

    # For Jan or OpenAI:
    endpoint_url: Optional[str] = None
    openai_api_key: Optional[str] = None  # for real openai calls

    # Common generation params:
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512
    truncation: bool = True
    temperature: float = 0.0


class BaseClaimGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        """
        Must return a list of lists, where each sub-list represents
        the claims extracted from the corresponding text in 'texts'.
        """
        raise NotImplementedError

    @staticmethod
    def build_claim_extraction_prompt(text: str) -> str:
        return REFINED_CLAIM_PROMPT.format(SOURCE_TEXT=text)

    @staticmethod
    def parse_json_output(output_str: str) -> List[str]:
        """
        Parse a single JSON string with a top-level "claims" field, e.g.:
           {"claims":["claim 1", "claim 2"]}
        Return the list of claims, or [] on error.
        """
        try:
            data = json.loads(output_str.strip())
            claims = data.get("claims", [])
            return claims if isinstance(claims, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i: i + size]


class BaseHuggingFaceGenerator(BaseClaimGenerator, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.tokenizer, self.model = self._load_model_and_tokenizer(config)

    def _load_model_and_tokenizer(self, config):
        # If you want dynamic import:
        if config.tokenizer_class and config.model_class:
            tokenizer_cls = self._import_class(config.tokenizer_class)
            model_cls = self._import_class(config.model_class)
        else:
            # fallback to AutoX
            tokenizer_cls = AutoTokenizer
            model_cls = AutoModel  # or AutoModelForSeq2SeqLM or something

        tokenizer = tokenizer_cls.from_pretrained(config.model_name_or_path)
        model = model_cls.from_pretrained(config.model_name_or_path)

        # Move to device
        model.to(self.device)
        return tokenizer, model

    @staticmethod
    def _import_class(class_path: str) -> Type:
        module_name, cls_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, cls_name)

    def _generate_single_batch(self, inputs):
        """Implement generation logic in child classes.
           Return list of raw strings (decoded)."""
        raise NotImplementedError


class HuggingFaceSeq2SeqGenerator(BaseHuggingFaceGenerator):
    def __init__(self, config):
        super().__init__(config)
        # If needed, re-initialize with the correct auto class
        if not isinstance(self.model, AutoModelForSeq2SeqLM):
            # Force correct model class:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path).to(self.device)

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        all_claims = []
        for batch in tqdm(self.chunked(texts, self.config.batch_size),
                          desc=f"Generating (Seq2Seq) with {self.config.model_name_or_path}"):
            # Prepare
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=self.config.truncation,
                max_length=self.config.max_length
            ).to(self.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,  # or a param you want
                temperature=self.config.temperature
            )

            # Decode
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for d in decoded:
                claims = self.parse_text_output(d)
                all_claims.append(claims)

        return all_claims

    @staticmethod
    def parse_text_output(output_str: str) -> List[str]:
        lines = [line.strip() for line in output_str.split(".") if line.strip()]
        return lines


class HuggingFaceCausalGenerator(BaseHuggingFaceGenerator):
    def __init__(self, config):
        super().__init__(config)
        # Force the correct model class if needed
        if not isinstance(self.model, AutoModelForCausalLM):
            self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path).to(self.device)

        # Many causal LMs need a pad token set:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        all_claims = []
        for batch in tqdm(self.chunked(texts, self.config.batch_size),
                          desc=f"Generating (Causal) with {self.config.model_name_or_path}"):
            # Build prompts
            prompts = [self.build_claim_extraction_prompt(t) for t in batch]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=self.config.truncation,
                max_length=self.config.max_length
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                do_sample=(self.config.temperature > 0.0)
            )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # DEBUG: Print out prompts and raw outputs
            for i, d in enumerate(decoded):
                print("\n--- DEBUG PROMPT ---")
                print(prompts[i])
                print("--- DEBUG MODEL OUTPUT ---")
                print(d)

                claims = self.parse_json_output(d)

                print("--- DEBUG CLAIMS AFTER PARSING ---")
                print(claims)

                all_claims.append(claims)

        return all_claims

    @staticmethod
    def parse_json_output(output_str: str) -> List[str]:
        """
        Extract the first JSON object containing a top-level "claims" key.
        Fix trailing commas before parsing.
        """
        print("Using regex parsing for LLaMa2 output.")
        pattern = r'(\{"claims"\s*:\s*\[.*?\]\})'
        match = re.search(pattern, output_str, flags=re.DOTALL)
        if not match:
            return []

        json_str = match.group(1)

        # Remove trailing commas before the closing bracket: ",]" => "]"
        json_str = re.sub(r',\s*\]', ']', json_str)

        try:
            data = json.loads(json_str)
            claims = data.get("claims", [])
            return claims if isinstance(claims, list) else []
        except json.JSONDecodeError:
            return []


class OpenAIClaimGenerator(BaseClaimGenerator):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.openai_api_key:
            openai.api_key = config.openai_api_key
        else:
            openai_api_key_env = os.getenv("OPENAI_API_KEY")
            if not openai_api_key_env:
                raise ValueError("OpenAI API key not found in config or in the environment.")
            openai.api_key = openai_api_key_env

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        all_claims = []
        for batch in tqdm(self.chunked(texts, self.config.batch_size),
                          desc=f"Generating (OpenAI) with {self.config.model_name_or_path}"):
            for text in batch:
                prompt = self.build_claim_extraction_prompt(text)

                # ChatCompletion call
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                content = response.choices[0].message.content
                claims = self.parse_json_output(content)
                all_claims.append(claims)

        return all_claims


class JanLocalClaimGenerator(BaseClaimGenerator):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        all_claims = []
        for batch in tqdm(self.chunked(texts, self.config.batch_size),
                          desc=f"Generating (Jan) with {self.config.model_name_or_path}"):
            for text in batch:
                prompt = self.build_claim_extraction_prompt(text)
                payload = {
                    "model": self.config.model_name_or_path,  # the engine name in Jan
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature
                }
                headers = {"Content-Type": "application/json"}

                resp = requests.post(self.config.endpoint_url, json=payload, headers=headers)
                resp.raise_for_status()

                data = resp.json()
                # for chat completions, we expect something like:
                # data["choices"][0]["message"]["content"]
                choices = data.get("choices", [])
                if not choices:
                    all_claims.append([])
                    continue

                content = choices[0]["message"]["content"]
                claims = self.parse_json_output(content)
                all_claims.append(claims)

        return all_claims
