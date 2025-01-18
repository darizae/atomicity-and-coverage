import importlib
from typing import List
from tqdm import tqdm


class BaseClaimGenerator:
    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        raise NotImplementedError("Subclasses must implement generate_claims")


class Seq2SeqClaimGenerator(BaseClaimGenerator):
    """
    A flexible claim generator that can handle different model architectures
    (T5, BART, etc.) based on config settings.
    """

    def __init__(
            self,
            model_name: str,
            tokenizer_class_path: str,
            model_class_path: str,
            device: str = "cpu",
            batch_size: int = 32,
            max_length: int = 512,
            truncation: bool = True
    ):
        """
        Initializes the claim generator with the specified model.

        Args:
            model_name (str): The Hugging Face Hub model name.
            tokenizer_class_path (str): Python path to the tokenizer class (e.g., transformers.T5Tokenizer).
            model_class_path (str): Python path to the model class (e.g., transformers.T5ForConditionalGeneration).
            device (str): Computation device ("cpu", "cuda", or "mps").
            batch_size (int): Batch size for processing.
            max_length (int): Maximum number of tokens for each input, if truncation is enabled.
            truncation (bool): Whether to truncate inputs exceeding max_length.
        """
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.truncation = truncation

        # Dynamically load the tokenizer class
        tokenizer_module_name, tokenizer_cls_name = tokenizer_class_path.rsplit(".", 1)
        tokenizer_module = importlib.import_module(tokenizer_module_name)
        tokenizer_cls = getattr(tokenizer_module, tokenizer_cls_name)

        # Dynamically load the model class
        model_module_name, model_cls_name = model_class_path.rsplit(".", 1)
        model_module = importlib.import_module(model_module_name)
        model_cls = getattr(model_module, model_cls_name)

        # Instantiate
        self.tokenizer = tokenizer_cls.from_pretrained(model_name)
        self.model = model_cls.from_pretrained(model_name).to(device)

    def generate_claims(self,
                        texts: List[str]
                        ) -> List[List[str]]:
        """
        Generate claims for a list of texts.

        Args:
            texts (List[str]): A list of input texts.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains generated claims for a text.
        """
        predictions = []
        for batch in tqdm(self._chunked(texts, self.batch_size),
                          desc=f"Generating claims with {self.model_name}"):
            # Tokenize with the configured max_length and truncation
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=self.truncation,
                max_length=self.max_length
            ).to(self.device)

            outputs = self.model.generate(**inputs)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # Simple sentence-level split. Adjust if you want more robust segmentation.
            claims = [text.split(". ") for text in decoded]
            predictions.extend(claims)
        return predictions

    @staticmethod
    def _chunked(iterable, size):
        """Yield successive n-sized chunks from an iterable."""
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]


class CausalLMClaimGenerator(BaseClaimGenerator):
    def __init__(self,
                 model_name: str,
                 tokenizer_class_path: str,
                 model_class_path: str,
                 device: str = "cpu",
                 batch_size: int = 4,  # smaller batch for big LLM
                 max_length: int = 1024):
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Dynamically load classes
        tokenizer_module_name, tokenizer_cls_name = tokenizer_class_path.rsplit(".", 1)
        tokenizer_module = importlib.import_module(tokenizer_module_name)
        tokenizer_cls = getattr(tokenizer_module, tokenizer_cls_name)

        model_module_name, model_cls_name = model_class_path.rsplit(".", 1)
        model_module = importlib.import_module(model_module_name)
        model_cls = getattr(model_module, model_cls_name)

        self.tokenizer = tokenizer_cls.from_pretrained(model_name)
        self.model = model_cls.from_pretrained(model_name)
        self.model.to(device)

    def generate_claims(self, texts: List[str]) -> List[List[str]]:
        predictions = []
        for batch in tqdm(self._chunked(texts, self.batch_size),
                          desc=f"Generating claims with {self.model_name} [CausalLM]"):
            # We'll create a prompt for each text to instruct the model
            prompts = [
                f"Extract atomic claims from the following text:\n\n{text}\n\nClaims:"
                for text in batch
            ]
            # Tokenize
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

            # Generate (we typically limit new tokens to avoid huge output)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,  # or True, depends on your needs
                eos_token_id=self.tokenizer.eos_token_id
            )

            decoded = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

            # Split the text by lines or by some delimiter to get claims
            # This is up to you, here's a naive approach:
            batch_claims = []
            for d in decoded:
                # remove the prompt portion if needed
                # naive approach: just split after "Claims:"
                # or do a more robust approach
                if "Claims:" in d:
                    d = d.split("Claims:")[-1]
                claims_list = [c.strip() for c in d.split("\n") if c.strip()]
                batch_claims.append(claims_list)

            predictions.extend(batch_claims)
        return predictions

    @staticmethod
    def _chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]
