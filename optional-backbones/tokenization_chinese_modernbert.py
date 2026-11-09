from typing import List, Optional, Union, Tuple
from transformers import PreTrainedTokenizerFast, BatchEncoding
from transformers.tokenization_utils_base import TruncationStrategy, PaddingStrategy
from transformers.utils import logging
from tokenizers.processors import TemplateProcessing
import re

logger = logging.get_logger(__name__)


class ChineseModernBertTokenizer(PreTrainedTokenizerFast):
    
    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]
    
    URL_PATTERN = re.compile(
        r'https?://[a-zA-Z0-9][a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]*',
        re.IGNORECASE
    )
    
    EMAIL_PATTERN = re.compile(
        r'\b[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}\b'
    )
    
    HTML_PATTERN = re.compile(r'<[^>]+>')
    
    CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    
    FULL_WIDTH_TO_HALF_WIDTH_TRANSLATOR = str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ　",
        "0123456789abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz "
    )
    
    MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n+')
    NON_NEWLINE_WHITESPACE_PATTERN = re.compile(r'[^\S\n]+')
    MULTIPLE_SPACES_PATTERN = re.compile(r' +')
    
    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        model_type: str = "bert",
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        do_text_preprocessing: bool = True,  # 新增参数：是否执行文本预处理
        **kwargs
    ):
        if model_type not in ["bert", "roberta"]:
            raise ValueError(f"model_type必须是'bert'或'roberta'，而不是'{model_type}'")
        
        self.model_type = model_type
        self.do_text_preprocessing = do_text_preprocessing
        
        super().__init__(
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )
        
        self._setup_post_processor()
    
    def _setup_post_processor(self):
        try:
            if self.model_type == "bert":
                if self.cls_token_id is None or self.sep_token_id is None:
                    logger.warning("Special token IDs not set, skipping post_processor setup")
                    return

                self._tokenizer.post_processor = TemplateProcessing(
                    single=f"{self.cls_token} $A {self.sep_token}",
                    pair=f"{self.cls_token} $A {self.sep_token} $B {self.sep_token}",
                    special_tokens=[
                        (self.cls_token, self.cls_token_id),
                        (self.sep_token, self.sep_token_id),
                    ],
                )
            elif self.model_type == "roberta":
                if self.bos_token_id is None or self.eos_token_id is None:
                    logger.warning("Special token IDs not set, skipping post_processor setup")
                    return
                
                self._tokenizer.post_processor = TemplateProcessing(
                    single=f"{self.bos_token} $A {self.eos_token}",
                    pair=f"{self.bos_token} $A {self.eos_token} {self.eos_token} $B {self.eos_token}",
                    special_tokens=[
                        (self.bos_token, self.bos_token_id),
                        (self.eos_token, self.eos_token_id),
                    ],
                )
        except Exception as e:
            logger.warning(f"Failed to setup post_processor: {e}")
            pass
    
    def _preprocess_text(self, text: str) -> str:

        if not isinstance(text, str) or not text:
            return ""

        text = self.EMAIL_PATTERN.sub("[EMAIL]", text)
        text = self.URL_PATTERN.sub("[URL]", text)

        text = self.HTML_PATTERN.sub('', text)

        text = text.translate(self.FULL_WIDTH_TO_HALF_WIDTH_TRANSLATOR)

        text = self.CONTROL_CHAR_PATTERN.sub('', text)

        text = self.MULTIPLE_NEWLINES_PATTERN.sub('\n', text)
        text = self.NON_NEWLINE_WHITESPACE_PATTERN.sub(' ', text)
        text = self.MULTIPLE_SPACES_PATTERN.sub(' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str, **kwargs) -> List[str]:

        if self.do_text_preprocessing:
            text = self._preprocess_text(text)
        
        return super().tokenize(text, **kwargs)
    
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
    ) -> BatchEncoding:

        if not is_split_into_words and self.do_text_preprocessing:
            processed_batch = []
            for item in batch_text_or_text_pairs:
                if isinstance(item, str):
                    processed_batch.append(self._preprocess_text(item))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    text, text_pair = item
                    processed_text = self._preprocess_text(text) if isinstance(text, str) else text
                    processed_pair = self._preprocess_text(text_pair) if isinstance(text_pair, str) else text_pair
                    processed_batch.append((processed_text, processed_pair))
                else:
                    processed_batch.append(item)
        else:
            processed_batch = batch_text_or_text_pairs
        
        return_token_type_ids = False
        
        return super()._batch_encode_plus(
            processed_batch,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            split_special_tokens=split_special_tokens,
        )
    
    def __call__(self, *args, **kwargs):

        if self._tokenizer.post_processor is None:
            self._setup_post_processor()
        
        return super().__call__(*args, **kwargs)
    
    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs
    ) -> str:

        text = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )
        
        text = text.replace(" ##", "").replace("##", "")
        
        return text
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:

        out_string = ""
        for i, token in enumerate(tokens):

            if token in self.all_special_tokens:

                if token == self.sep_token:
                    out_string += f" {token} "
                else:
                    out_string += token + " "

            elif token.startswith("##"):
                out_string += token[2:]

            else:
                if i > 0 and not tokens[i - 1] in self.all_special_tokens:
                    out_string += " "
                out_string += token
        
        return out_string.strip()
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:

        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)