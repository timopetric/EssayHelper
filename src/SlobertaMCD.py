import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.tokenization_utils_base import BatchEncoding
import numpy as np
from tqdm import tqdm
import re
from helpers import *
from datetime import datetime
from typing import List, Union
from pprint import pprint
from constants import *


def set_dropout_to_train(m):
    if type(m) == torch.nn.Dropout:
        m.train()

class SlobertaMCD:
    def __init__(self, tokenizer=None,
                       verbose=True,
                       re_tokenizer_filter_str=RE_DEFAULT_TOKEN_STR_FILTER,
                       cuda_device=-1,
                       huggingface_online=False) -> None:

        debug_dt_init_start = datetime.now()
        self.version = "0.1.0"

        self.huggingface_online = huggingface_online
        self.verbose = verbose
        cuda_str = "cuda" + f":{cuda_device}" if cuda_device > -1 else ""
        self.device = torch.device(cuda_str if cuda_str and torch.cuda.is_available() else "cpu")

        self.re_tokenizer_filter = re.compile(re_tokenizer_filter_str) if re_tokenizer_filter_str else None

        try:
            self.tokenizer, self.model = self.load_model_and_tokenizer(tokenizer, self.huggingface_online)
        except EnvironmentError:
            print("On first run the huggingface online param should be set to True so that the model can be downloaded. Retrying online.")
            self.tokenizer, self.model = self.load_model_and_tokenizer(tokenizer, True)

        self.model = self.model.to(self.device)

        self.decode_indeces_to_token_strs = self.init_tokenizer_decode_vectorized_np_func()
        self.translate_indeces = self.init_indeces_translation_np_func()

        self.tokenizer_num_tokens = len(self.tokenizer.get_vocab())
        self.token_inxs, self.token_strs, self.token_inxs_mask = self.init_token_lists(self.tokenizer_num_tokens, self.decode_indeces_to_token_strs)
        
        if self.verbose:
            debug_dt_init_end = datetime.now()
            self.print_config(debug_dt_init_start, debug_dt_init_end)

    def load_model_and_tokenizer(self, tokenizer, huggingface_online):
        tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained("EMBEDDIA/sloberta", local_files_only=not huggingface_online)
        model = AutoModelForMaskedLM.from_pretrained("EMBEDDIA/sloberta", local_files_only=not huggingface_online)
        return tokenizer, model

    def print_config(self, debug_dt_init_start, debug_dt_init_end):
        txt = f"Using device: {self.device}\n"
        txt += f"Init and model loading took: {debug_dt_init_end-debug_dt_init_start}"
        print(txt)


    def ensure_exactly_one_mask_token(self, masked_index:torch.Tensor):
        numel = masked_index.numel()
        if numel > 1:
            raise ValueError(
                "fill-mask",
                self.model.base_model_prefix,
                f"More than one mask_token ({self.tokenizer.mask_token}) is not supported",
            )
        elif numel < 1:
            raise ValueError(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )

    def init_token_lists(self, tokenizer_num_tokens, decode_indeces_to_token_strs):
        """
        returns 3 lists 
            - token_inxs      : list of indeces
            - token_strs      : list of tokens strings from the tokenizer dictionary
            - token_inxs_mask : list of 
        """
        token_inxs = list(range(0, tokenizer_num_tokens, 1))
        token_strs = decode_indeces_to_token_strs(token_inxs).tolist()
        assert token_strs[self.tokenizer.mask_token_id] == self.tokenizer.mask_token, "token_inxs seem to be incorrectly converted to token_strs"
        assert len(token_inxs) == len(token_strs)

        token_inxs_mask = list()
        for token_inx, token_str in zip(token_inxs, token_strs):
            if self.re_tokenizer_filter is None:
                token_inxs_mask.append(token_inx)   # append all tokens  
            elif self.re_tokenizer_filter.fullmatch(token_str):
                # print("adding:", token_inx, token_str)
                token_inxs_mask.append(token_inx)

        if self.verbose:
            print(f"Tokenizer vocabulary had {tokenizer_num_tokens} tokens and was shrinked to: {len(token_inxs_mask)} tokens.")

        return token_inxs, token_strs, token_inxs_mask


    def __call__(self, *args, **kwargs):
        return self.pipeline_process(*args, **kwargs)

    def init_tokenizer_decode_vectorized_np_func(self):
        return np.vectorize(lambda x: self.tokenizer.decode(int(x)))

    def init_indeces_translation_np_func(self):
        # vocabulary has 32005 tokens, whe select only a subset of those (lets say 30000)
        # this function translates new indeces (from 0 to 29999) and returns the correct indeces in vocabulary
        # so that words are selected correctly
        # (this is usefull because sorting on a GPU is faster)
        return np.vectorize(lambda i: self.token_inxs_mask[i])

    def tokenize_batch(self, input_batch):
        """
        input_batch : list of strings (sentences)
        """

        if isinstance(input_batch, BatchEncoding):
            inputs = input_batch
        elif isinstance(input_batch, list) and all([isinstance(i, str) for i in input_batch]):
            inputs = self.tokenizer(
                input_batch,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation="do_not_truncate",
            )
        else:
            raise ValueError(f"Input batch is not in correct form. Should either be a pretokenized obj of type "
                             f"BatchEncoding (AutoTokenizer) or a list of raw sentence strings.")

        inputs_gpu = {  # put inputs to gpu
            name: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in inputs.items()
        }
        return inputs, inputs_gpu

    def pipeline_process(self, input_sentences: Union[BatchEncoding, List[str]],
                   dropouts_enable:bool=True,
                   topk:int=10,
                   num_passes:int=50,
                   verbose:bool=False):
        assert (dropouts_enable and num_passes > 1) or (not dropouts_enable and num_passes == 1)

        if verbose:
            if dropouts_enable:
                print(f"Using dropout enabled averaging over {num_passes} passes.\n")
            else:
                print(f"Only performing one normal pass.\n")

        # load and tokenize input sentences
        inputs, inputs_gpu = self.tokenize_batch(input_sentences)
        batch_size_inputs = inputs["input_ids"].size(0)
        
        self.model.eval()
        if dropouts_enable:
            self.model.apply(set_dropout_to_train) # activate droupout levels

        # 1dim tensor of indeces of mask tokens in each sentence in the input tbatch
        sent_mask_inxs = list()
        for i in range(batch_size_inputs):
            input_ids = inputs["input_ids"][i]
            masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze()
            self.ensure_exactly_one_mask_token(masked_index)
            sent_mask_inxs.append(masked_index)
        sent_mask_inxs = torch.as_tensor(sent_mask_inxs).to(self.device)

        # 1dim tensor of indeces of all tokens (from 0 to batch_size_inputs)
        tensor_dummy_range_1_to_batch_size = torch.arange(batch_size_inputs).to(self.device)

        # 1dim tensor filler to apply slicing to whole 1 dimension of outputs (from 0 to batch_size_inputs)
        tensor_tokens_inxs_mask = torch.as_tensor(self.token_inxs_mask).to(self.device)

        # 2d tensor of sums
        tensor_avgs = torch.zeros(batch_size_inputs, len(self.token_inxs_mask), dtype=torch.float64).to(self.device)
        # tensor_avgs = torch.zeros(32005).to(self.device)   # TODO: added 

        dt_model_start = datetime.now()
        # num_passes = 1 if not dropouts_enable else num_passes
        assert (dropouts_enable and num_passes > 1) or (not dropouts_enable and num_passes == 1)
        for i in tqdm(range(num_passes), desc=f"Getting {num_passes} preds for {batch_size_inputs} sents", disable=not verbose):
            with torch.no_grad():   # do not save gradients
                outputs = self.model(**inputs_gpu)[0]
                outputs = outputs[tensor_dummy_range_1_to_batch_size, sent_mask_inxs, :]
                outputs = torch.index_select(outputs, 1, tensor_tokens_inxs_mask)

                # calculate softmaxes for batch outputs and add them to tmp all sum tensor
                outputs_softmax = torch.softmax(outputs.T, dim=0).T
                tensor_avgs = torch.add(tensor_avgs, outputs_softmax)

        # divide to get softmax averages
        if num_passes > 1:
            tensor_avgs = torch.div(tensor_avgs, num_passes)
        assert 0.999 < torch.sum(tensor_avgs[0,:]) < 1.001, "Softmax was not applied correctly"
        
        # sort tokens by score and put the topk results to the cpu
        avgs_sorted_vals, avgs_sorted_inxs = torch.sort(tensor_avgs, 1, descending=True)
        avgs_sorted_vals = avgs_sorted_vals[:, :topk].cpu().numpy()
        avgs_sorted_inxs = avgs_sorted_inxs[:, :topk].cpu().numpy()
        assert avgs_sorted_vals[0, 0] > avgs_sorted_vals[0, 1] > avgs_sorted_vals[0, 2], "Sorting was done incorrectly"
        assert avgs_sorted_vals.shape[0] == batch_size_inputs, "Number of outputs is not the same as num of inputs in the batch"

        if verbose:
            dt_model_end = datetime.now()
            print(f"Batch processing of {batch_size_inputs} input sents with {num_passes} passes took: {dt_model_end - dt_model_start}")
        dt_postproc_start = datetime.now()

        # pack results
        results = list()
        for i in range(avgs_sorted_vals.shape[0]):
            token_inxs = avgs_sorted_inxs[i]
            inx2 = self.translate_indeces(token_inxs)
            token_strs3 = self.decode_indeces_to_token_strs(inx2)

            one_input = list()
            for j in range(avgs_sorted_vals.shape[1]):
                # TODO: possible optimization: return a list of named tuples
                one_input.append(
                    {
                        "score": avgs_sorted_vals[i][j],
                        "token": token_inxs[j],
                        "token_str": token_strs3[j],
                    }
                )
            results.append(one_input)

        if verbose:
            dt_postproc_end = datetime.now()
            print(f"Postprocessing with batch size {batch_size_inputs} took: {dt_postproc_end - dt_postproc_start}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results


def compare_pipelines_and_get_sentence_predictions(sent_list: List[str], enable_huggingface_pipline=True):
    outs = list()
    smcd = SlobertaMCD(cuda_device=1, verbose=True) #, re_tokenizer_filter_str=r".*?")

    num_passes_list = []

    if enable_huggingface_pipline:
        from transformers import pipeline
        tokenizer_sloberta = smcd.tokenizer
        model_sloberta = smcd.model
        nlp_sloberta = pipeline('fill-mask', model=model_sloberta, tokenizer=tokenizer_sloberta, top_k=20, device=1)
        outs.append(nlp_sloberta(sent_list))
        num_passes_list += [1]


    outs.append(smcd(sent_list, dropouts_enable=False, topk=10, num_passes=1))
    num_passes_list += [1]


    num_passes_list_tmp = list(range(10, 120, 30))
    for num_passes in num_passes_list_tmp:
        out = smcd(sent_list, dropouts_enable=True, topk=10, num_passes=num_passes)
        outs.append(out)
    num_passes_list += num_passes_list_tmp

    for i in range(len(sent_list)):
        print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("$$$$$$$$$$", sent_list[i], "$$$$$$$$$$$$$")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        
        for j in range(len(out[i])):
            strs = [x[i][j].get('token_str') for x in outs]
            scores = [x[i][j].get('score') for x in outs]
            if j == 0:
                for k in range(len(strs)):
                    print(f"# PASSES: {list(num_passes_list)[k]:<14d}", end="\t")
                print()
            for k in range(len(strs)):
                print(f"{scores[k]:.5f} : {strs[k]:14s}", end="\t")
            print()




def main():
    sent_list = [
        "<mask> bi, ampak tega ni storil.",
        "Lahko <mask>, ampak tega ni storil.",
        "Lahko bi, <mask> tega ni storil.",
        # "Lahko bi, ampak <mask> ni storil.",
        # "On je hotel <mask> ampak ni šel.",
        "On je hotel <mask> ampak ni šel tja kamor bi lahko odšel.",
        "Želela je samo <mask>.",
        "Lahko bi, ampak tega <mask> storil.",
        "Lahko bi, ampak tega ni <mask>.",
        ]
    # sent_list2 = [sent_list[i%len(sent_list)] for i in range(570)]
    # sent_list2 = [sent_list[4] for i in range(100)]
    # sent_list2 = sent_list[:1]
    compare_pipelines_and_get_sentence_predictions(sent_list)


if __name__ == "__main__":
    main()