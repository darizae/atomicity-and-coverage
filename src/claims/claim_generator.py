from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from typing import List


class ClaimGenerator:
    """
    A modular claim generator that supports various models for claim generation.
    """
    def __init__(self,
                 model_name: str,
                 device: str = "cpu",
                 batch_size: int = 32
                 ):
        """
        Initializes the claim generator with the specified model.

        Args:
            model_name (str): Name of the model from the Hugging Face Hub.
            device (str): Device to run the model on ("cpu", "cuda", or "mps").
            batch_size (int): Batch size for processing claims.
        """
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

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
        for batch in tqdm(self._chunked(texts, self.batch_size), desc=f"Generating claims with {self.model_name}"):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            claims = [text.split(". ") for text in decoded]  # Simple sentence splitting
            predictions.extend(claims)
        return predictions

    @staticmethod
    def _chunked(iterable, size):
        """Yield successive n-sized chunks from an iterable."""
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]
