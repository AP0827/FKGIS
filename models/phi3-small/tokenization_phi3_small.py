# Adapted from https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/tokenization_qwen.py
import os
from typing import Collection, List, Optional, Dict, Set, Tuple, Union

from functools import cached_property

import base64
import requests

from transformers import PreTrainedTokenizer, AddedToken, AutoConfig
from transformers.models.auto.tokenization_auto import get_tokenizer_config
import tiktoken


"""
    This tokenizer is almost identical to tiktoken.get_encoding("cl100k_base")
    with a few additional special tokens to support the ChatML format.

    TODO(bapatra): Right now, I do not save the special tokens to the vocab file.
    Maybe in the future, that would be useful? Can add that support later.

"""

def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }

# On the megatron codebase, we pad vocabularies to ensure matrix multiplication is fast.
# this in turn causes some indices to be empty. We account for these empty indices by adding
# dummy tokens to the tokenizer.

EFFECTIVE_PADDED_VOCAB_SIZE = 100352
ACTUAL_VOCAB_SIZE = 100276


DUMMY_TOKENS = {
    f"<|dummy_id_{11 + offset}|>": 100276 + offset
    for offset in range(1, EFFECTIVE_PADDED_VOCAB_SIZE - ACTUAL_VOCAB_SIZE)
}

SPECIAL_TOKENS = {
    # tiktoken.get_encoding("cl100k_base")._special_tokens
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    # Special tokens for post-training
    "<|system|>": 100261, 
    "<|user|>": 100262,
    "<|assistant|>": 100263,
    # Dummy unused tokens
    "<|dummy_id_0|>": 100264,
    "<|dummy_id_1|>": 100265,
    # Special tokens for post-training continued
    "<|end|>": 100266,
    # Some dummy tokens, so that tokenization is contiguous and does not cause issues
    # Note that the 100256th token of tiktoken.get_encoding("cl100k_base") does not
    # actually map to anything. So we use a dummy token here.
    "<|dummy_id_2|>": 100256,
    # Likewise, tokens from 100267 to 100275 are also unused
    "<|dummy_id_3|>": 100267,
    "<|dummy_id_4|>": 100268,
    "<|dummy_id_5|>": 100269,
    "<|dummy_id_6|>": 100270,
    "<|dummy_id_7|>": 100271,
    "<|dummy_id_8|>": 100272,
    "<|dummy_id_9|>": 100273,
    "<|dummy_id_10|>": 100274,
    "<|dummy_id_11|>": 100275,
    # The final end of prompt token
    # (unused, but present as a part of tiktoken.get_encoding("cl100k_base")._special_tokens)
    '<|endofprompt|>': 100276,
    # Dummy tokens to account for padding of the tokenizer
    # We pad to ensure tensor cores are used for vocab multiplication
    **DUMMY_TOKENS
}

class Phi3SmallTokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "cl100k_base.tiktoken"
    }

    model_input_names: List[str] = ["input_ids", "attention_mask"]
    padding_side = "left"

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        errors: str = "replace",
        **kwargs
    ) -> None:
        # PreTrainedTokenizer's init calls _add_tokens, which in turn checks
        # if the token is present in `self.special_tokens``. Hence instantiating it here.
        # The way Qwen gets around this is by checking against SPECIAL_TOKENS
        # But I think it's better to check against the objects own `special_tokens`
        # in case we eventually want to allow the tokenizer to have special tokens.
        self.special_tokens = SPECIAL_TOKENS

        super().__init__(**kwargs)
        self.errors = errors

        try:
            base = tiktoken.get_encoding("cl100k_base")
        # This deals with the scenario where user has restricted internet access 
        # and thus fails to download the tokenizer file from https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
        # It is assumed that user should be able to access files on huggingface hub.
        except requests.RequestException:
            import hashlib
            from transformers.utils import cached_file
            cached_tokenizer_path = cached_file(
                    "microsoft/Phi-3-small-8k-instruct",
                    "cl100k_base.tiktoken",
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False
                )
            tiktoken_cache_dir = os.path.dirname(cached_tokenizer_path)
            tiktoken_cache_path = os.path.join(
                tiktoken_cache_dir, 
                hashlib.sha1("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken".encode()).hexdigest()
            )
            if not os.path.exists(tiktoken_cache_path):
                os.rename(cached_tokenizer_path, tiktoken_cache_path)
            os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
            base = tiktoken.get_encoding("cl100k_base")
            
        if vocab_file is None:
            self.mergeable_ranks: Dict[bytes, int] = base._mergeable_ranks
        else:
            self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)

        self.pat_str = base._pat_str
        
        enc = tiktoken.Encoding(
            name="phi3small",
            pat_str=self.pat_str,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc

        self.decoder: Dict[int, bytes] = {
            v: k for k, v in self.mergeable_ranks.items()
        }
        self.decoder.update({v: k for k, v in self.special_tokens.items()})
        
        self.eod_id = self.tokenizer.eot_token
        self._eos_token = self._convert_id_to_token(self.eod_id)

        # Setting the bos_token to be the same as the eos_token
        # Note that this is **not** the correct thing to do, and is done
        # just so that some of the downstream libraries do not break.
        self._bos_token = self._eos_token

        # Assign the special tokens to class variables
        self.system_id = self.special_tokens["<|system|>"]
        self.user_id = self.special_tokens["<|user|>"]
        self.assistant_id = self.special_tokens["<|assistant|>"]
        self.end_id = self.special_tokens["<|end|>"]
    
    @cached_property
    def dummy_token_indices(self) -> List[int]:
        # There are some additional special tokens in the cl100k_base tokenizer
        # that we do not use. Hence, we also consider them to be dummy tokens.
        additional_tokens = [
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "<|endofprompt|>"
        ]
        dummy_token_indices = [index for token, index in self.special_tokens.items() if "dummy_id" in token]
        dummy_token_indices.extend([self.special_tokens[token] for token in additional_tokens])
        return sorted(dummy_token_indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["tokenizer"]
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        enc = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=self.pat_str,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.tokenizer = enc
    
    def __len__(self):
        return self.tokenizer.n_vocab
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        **kwargs,
    ):
        cls_kwargs = kwargs
        # First try to load from the tokenization config if it exists
        tokenization_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        if tokenization_config:
            cls_kwargs = {
                **tokenization_config,
                **cls_kwargs
            }
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            cls_kwargs["model_max_length"] = config.max_position_embeddings
        return cls(**cls_kwargs)

    def get_vocab(self) -> Dict[Union[str, bytes], int]:
        return {**self.mergeable_ranks, **self.special_tokens}
    
    def convert_tokens_to_ids(
        self,
        tokens: Union[bytes, str, List[Union[bytes, str]]]
    ) -> Union[int, List[int]]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        ids: List[int] = []
        for token in tokens:
            ids.append(self.convert_tokens_to_ids(token))
        return ids

    def _add_tokens(
            self,
            new_tokens: Union[List[str], List[AddedToken]],
            special_tokens: bool = False,
    ) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Only special tokens can be added to this tokenizer")
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in self.special_tokens:
                raise ValueError(
                    "For now, we do not support unknown special tokens\n"
                    "In the future, if there is a need for this, we can add special tokens to the tokenizer\n"
                    "starting from rank 100261 - 100263 and then 100266 - 100275.\n"
                    "And finally, we can re-construct the enc object back\n"
                )
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        file_path = os.path.join(save_directory, "cl100k_base.tiktoken")
        with open(file_path, "w") as f:
            for token, rank in self.mergeable_ranks.items():
                line = base64.b64encode(token).decode("utf-8") + " " + str(rank) + "\n"
                f.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs
    ) -> List[Union[bytes, str]]:
        tokens: List[Union[bytes, str]] = []
        for token_id in self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[token_id])
        return tokens

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    @property
    def eos_token_id(self) -> int:
        return self.eod_id

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).
        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)


