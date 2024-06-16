import torch
from typing import Tuple
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer


def download_embedding_configs(model_name: str, model_path: str, tokenizer_path: str) -> bool:
    """
    Function to download embedding model and its tokenizer from huggingface hub

    :Parameters:
    model_name: Name of the embedding model to download.
    model_path: Absolute path where the embedding model will be stored.
    tokenzier_path: Absolute path where the embedding model tokenizer will be stored.

    :Returns:
    Boolean value representing the completion of the process
    """

    # Load embedding model and its tokenizer
    embedding_model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Save the model and tokenizer
    torch.save(embedding_model, model_path)
    tokenizer.save_pretrained(tokenizer_path)

    return True


def get_embedding_config(model_path: str, tokenizer_path: str) -> Tuple[torch.nn.Module, Tokenizer]:
    """
    Function to load locally saved embedding model and its tokenizer.

    :Parameters:
    model_path: Absolute path to the locally saved embedding model.
    tokenizer_path: Absolute path to the embedding model's tokenizer.

    :Returns:
    Embedding model and its tokenizer
    """

    embedding_model = torch.load(model_path)
    embedding_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return embedding_model, embedding_tokenizer