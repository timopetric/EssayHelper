from tkinter import NONE
import torch
import classla
from classla.models.common.doc import Document as DocumentClassla
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from curtsies.fmtfuncs import green, bold, yellow, red
from tqdm import tqdm
import toma
import re

from SlobertaMCD import SlobertaMCD
from WordMsd import WordMsd
from SentenceHelper import SentenceHelper
from DictHelper import DictHelper
from constants import *
from thresholds import *

import copy
import itertools
from pprint import pformat
from typing import Dict, Generator, Iterable, List, Tuple
from datetime import datetime
import sherpa 
from helpers import EvalParams, ConfusionMatrix
import logging

from helpers import Thresholds
logger = logging.getLogger("__main__." + __name__)


class EssayHelper:
    def __init__(self,
            cl_text_original: str,
            cuda_device=0,
            verbose=True,
            cl_pipeline: classla.Pipeline=None,
            cl_pipeline_pretok: classla.Pipeline=None,
            tokenizer_sloberta=None,
            sloberta_mcd: SlobertaMCD=None,
            huggingface_online=False,
            tqdm_enable=True,
            eval_params: EvalParams=None,
            thresholds: Thresholds=None,
            multi_pass = True,
            sloberta_kwargs = None,
            sherpa_optimizations_todo=[],
            mcd_on = False,
            refresh_dict_helper=False,
            thresholds_prelearned_option = 0  # 0 = both, 1 = case, 2 = number
            ):

        self.huggingface_online = huggingface_online
        self.tqdm_enable = tqdm_enable
        self.eval_params: EvalParams = eval_params
        self.verbose = verbose
        self.multi_pass = multi_pass

        debug_dt_pipelines_and_dh_load = datetime.now()
        use_gpu = cuda_device > -1
        self.cl_pipeline, self.cl_pipeline_pretok = self.init_classla_pipeline(cl_pipeline, cl_pipeline_pretok, use_gpu=use_gpu, verbose=False)
        self.nlp_sloberta, self.tokenizer_sloberta = self.init_sloberta_mcd_pipeline(sloberta_mcd, tokenizer_sloberta, use_gpu=use_gpu, cuda_device=cuda_device, huggingface_online=self.huggingface_online)
        self.dh: DictHelper = DictHelper(refresh=refresh_dict_helper)
        if self.verbose: logger.info(f"Pipelines and dh init took: {datetime.now()-debug_dt_pipelines_and_dh_load}")

        self.tokens_batch_max_sloberta = 5000
        self.tokens_batch_max_classla = 5000        
        self.sloberta_kwargs = sloberta_kwargs if sloberta_kwargs else {"topk": 10, "num_passes": 50, "dropouts_enable": True} if mcd_on else {"topk": 10, "num_passes": 1, "dropouts_enable": False}

        self.skip_unclassified = True

        self.thresholds = thresholds if thresholds else get_thresholds(thresholds_prelearned_option, mcd_on)
        print(f"Essay thresholds.case42:   {self.thresholds.case42}")
        print(f"Essay thresholds.number32: {self.thresholds.number32}")

        self.sherpa_optimizations_todo = sherpa_optimizations_todo # ["classla_batch", "sloberta_batch"]
        self._sherpa_optimizations_was_not_empty = len(self.sherpa_optimizations_todo) > 0

        self.cl_text_original = cl_text_original
        self.sentences: List[SentenceHelper] = []
        self._classla_process_orig_text()


    def init_classla_pipeline(self, cl_pipeline: classla.Pipeline, cl_pipeline_pretok: classla.Pipeline, use_gpu: bool, verbose: bool=False):
        # TODO: is it possible to select a specific cuda gpu
        if self.verbose: logger.info(f"Loading classla pipeline {'to the' if use_gpu else 'without'} gpu.")
        cl_pipeline = cl_pipeline if cl_pipeline is not None else classla.Pipeline('sl', processors="tokenize,pos,lemma", use_gpu=use_gpu, verbose=verbose)

        # TODO: init outside and pass here?
        cl_pipeline_pretok = cl_pipeline_pretok if cl_pipeline_pretok is not None else classla.Pipeline('sl', processors="tokenize,pos,lemma", tokenize_pretokenized=True, use_gpu=use_gpu, verbose=verbose)

        return cl_pipeline, cl_pipeline_pretok

    def init_sloberta_mcd_pipeline(self, sloberta_mcd: SlobertaMCD, tokenizer_sloberta, use_gpu: bool, cuda_device, huggingface_online=False):
        if self.verbose: logger.info(f"Loading SloBERTa pipeline {'to the' if use_gpu else 'without'} gpu.")

        if tokenizer_sloberta is None:
            # type: transformers.models.camembert.tokenization_camembert_fast.CamembertTokenizerFast
            try:
                tokenizer_sloberta = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta", local_files_only=not huggingface_online, verbose=False)
            except EnvironmentError:
                print("On first run the huggingface online param should be set to True so that the model can be downloaded. Retrying online.")
                tokenizer_sloberta = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta", local_files_only=False, verbose=False)

        sloberta_mcd = sloberta_mcd if sloberta_mcd is not None else SlobertaMCD(tokenizer=tokenizer_sloberta, cuda_device=cuda_device if use_gpu else -1, verbose=False)
        return sloberta_mcd, tokenizer_sloberta

    def destroy_all_pipelines(self):
        self.cl_pipeline = None
        self.cl_pipeline_pretok = None
        self.nlp_sloberta = None
        self.tokenizer_sloberta = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Freed all pipelines from memory.")

    def destroy_cl_pipeline(self):
        self.cl_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Freed cl_pipeline from memory.")

    def run_optimization(self, option, settings, args=None, exit_if_done=True):
        if option in self.sherpa_optimizations_todo:
            self.sherpa_optimizations_todo.remove(option)
            self._run_sherpa_optimization(
                option,
                settings,
                args,
            )
        if self._sherpa_optimizations_was_not_empty and exit_if_done and not self.sherpa_optimizations_todo:
            exit()

    def _classla_process_orig_text(self):
        ######### RUN CLASSLA PIPELINE TO TOKENIZE ALL SENTENCES AND WORDS #########
        debug_dt_classla_start = datetime.now()
        doc: DocumentClassla = self.cl_pipeline(self.cl_text_original)
        # self.destroy_cl_pipeline()
        if self.verbose: logger.info(f"Initial classla pipeline processing took: {datetime.now()-debug_dt_classla_start}")


        ######### BUILD THE SENTENCE MODELS FROM CLASSLA RETURN #########
        debug_dt_sentences_init_start = datetime.now()
        for sentence in doc.sentences:
            sent = SentenceHelper(sentence, self.dh, self.eval_params, self.thresholds)
            self.sentences.append(sent)
        if self.verbose: logger.info(f"Initial essay sentences init took: {datetime.now()-debug_dt_sentences_init_start}")

        ######### DEBUG PRINT SENTENCES WITH WRONG SET WORDS #########
        if self.verbose and self.eval_mode:
            print("Sentences with wrong words:")
            for s in self.eval_iter_sentences_with_wrong_words_set():
                print(f"\t{s.text_orig_and_wrong_colored}")


        ######### POTENTIALLY RUN SLOBERTA BATCH PARAMETER OPTIMIZATION #########
        self.run_optimization("sloberta_batch", {"batch_from": 200, "batch_to":2600, "max_searches": 100, "algorithm": "random"})


        ######### BATCH RUN SLOBERTA PIPELINE TO GET WORD SUGGESTIONS FOR EVERY POSSIBLY MASKED WORD THERE IS IN THE DOCUMENT #########
        self.tokens_batch_max_sloberta = 1142
        debug_dt_sloberta_start = datetime.now()
        sloberta_out, sloberta_repl_word_probabilities_list, sloberta_out_len = toma.simple.batch(self.run_sloberta_batched_pipeline, self.tokens_batch_max_sloberta, tqdm_enable=self.tqdm_enable)
        # sloberta_out, sloberta_out_len = toma.simple.range(self.run_sloberta_pipeline, 5000, 20000, 100, self.tokens_batch_max_sloberta, sloberta_topk)
        # print(sloberta_out)
        # # [[{'score': 0.5776349630366349, 'token': 372, 'token_str': 'tega'},
        # #   {'score': 0.027967909684584954, 'token': 5771, 'token_str': 'ničesar'},
        # #   {'score': 0.0228437853414016, 'token': 1277, 'token_str': 'nič'}],
        # #  [{'score': 0.6115105530619621, 'token': 1805, 'token_str': 'nisem'},
        # #   {'score': 0.12968991036061198, 'token': 95, 'token_str': 'ni'},
        # #   {'score': 0.07968529211822897, 'token': 13708, 'token_str': 'nebi'}], ...]
        if self.verbose: logger.info(f"Sloberta processing took: {datetime.now()-debug_dt_sloberta_start} for {sloberta_out_len} sentences.")


        ######### POTENTIALLY RUN CLASSLA BATCH PARAMETER OPTIMIZATION #########
        self.run_optimization("classla_batch", {"batch_from": 200, "batch_to": 200000, "max_searches": 200, "algorithm": "grid"}, [sloberta_out, sloberta_out_len])


        ######### BATCH RUN CLASSLA PIPELINE TO CLASSIFY EVERY SLOBERTA SUGGESTED REPLACEMENT WORD #########
        self.tokens_batch_max_classla = 16000
        debug_dt_fake_word_classifications_start = datetime.now()
        classla_out = toma.simple.batch(self.run_classla_batched_pipeline, self.tokens_batch_max_classla, sloberta_out, sloberta_out_len)
        assert len(classla_out) == len(sloberta_repl_word_probabilities_list), \
            f"The number of sentences in classla_out ({len(classla_out)}) and number of replacement words ({sloberta_repl_word_probabilities_list}) should be the same."
        if self.verbose: logger.info(f"Classla round 2 took: {datetime.now()-debug_dt_fake_word_classifications_start}")


        ######### SAVE THE MASKED SENTENCE SLOBERTA REPLACEMENT WORDS IN THE MODELS #########
        self.process_classla_replacement_words(classla_out, sloberta_repl_word_probabilities_list, multi_pass=False)


        ######### calculate word form predictions if not in eval mode #########
        if not self.eval_mode:
            self.set_predictions(self.thresholds, multiple_passes=self.multi_pass, print_verbose_all_words=self.verbose)


        # EVAL AFTER PREDICTIONS ARE DONE
        # if self.eval_mode:
        #     self.eval_analisys()
        #     for s in self.eval_iter_predicted_sentences():
        #         print(s.text_eval_orig_corrected_colored)        


        if self.verbose: logger.info(f"$$$$$$ Finished. Essay processing took: {datetime.now()-debug_dt_classla_start} $$$$$$")
        
        if not self.eval_mode:
            print()
            print(self)

        ######### DEBUG PRINT SENTENCES AND REPLACEMENT WORDS #########
        # print(self.text_orig_and_eval_corrected_colored)
        
        # for sent in self.sentences:
        #     for w in sent.words:
        #         print(f"{w.text: <20s} : {[f'{i.text} ({i.sloberta_certainty:.3f})' for i in w.word_replacements]}")

        # print(self.verbose_print())
    
    def eval_analisys(self, verbose=True):
        num_sentences = 0
        num_wrong_words = 0
        num_correct_replacements = 0
        num_incorrect_replacements = 0
        
        num_all_words = 0
        
        tp, fp, tn, fn = 0, 0, 0, 0
        re_match_slo_words = re.compile(RE_DEFAULT_TOKEN_STR_FILTER_1)
        

        for s in self.sentences:
            num_sentences += 1

            sentence_corrected_correctly = True
            for w in s.words:
                
                if re_match_slo_words.fullmatch(w.text):
                    num_all_words += 1

                if w.has_wrong_word:
                    num_wrong_words += 1

                    if w.has_correct_repl:
                        tp += 1
                    else:
                        fn += 1

                else:
                    if w.word_selected_str and w.word_selected_str != w.text:
                        fp += 1
                    else:
                        if re_match_slo_words.fullmatch(w.text):
                            tn += 1
                    
                if w.has_correct_repl:
                    num_correct_replacements += 1
                
                else:
                    num_incorrect_replacements += 1
                    sentence_corrected_correctly = False
            
            if verbose and not sentence_corrected_correctly:
                print(s.text_eval_orig_corrected_colored)
                for w in s.words:
                    if w.eval_has_new_replacement_w or w.has_wrong_word:
                        print(w.verbose_string())
                print()

        conf_matrix = ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)

        return num_sentences, num_wrong_words, num_correct_replacements, conf_matrix, num_all_words
    
    
    def eval_iter_sentences_with_wrong_words_set(self) -> Iterable[SentenceHelper]:
        # self._eval_check_if_sentence_predictions_set()
        for s in self.sentences:
            if any([w.has_wrong_word for w in s.words]):
                yield s
                
    def eval_iter_sentences_with_wrong_words_and_new_predictions(self) -> Iterable[SentenceHelper]:
        for s in self.sentences:
            for w in s.words:
                if w.has_wrong_word and w.has_new_replacement_w:
                    yield s
                    continue
                
    def eval_iter_predicted_sentences(self) -> Iterable[SentenceHelper]:
        self._eval_check_if_sentence_predictions_set()
        for s in self.sentences:
            if s.eval_predictions_set:
                yield s

    def _eval_check_if_sentence_predictions_set(self):
        for s in self.sentences:
            s._eval_check_if_predictions_set()

    def _run_sherpa_optimization(self, option, settings, args):
        assert option in {"sloberta_batch", "classla_batch"}
        assert isinstance(settings, dict)
        assert all([i in {"batch_from", "batch_to", "max_searches", "algorithm"} for i in settings])
        assert all([isinstance(settings[i], int)  for i in ["batch_from", "batch_to", "max_searches"]])

        print(f"Doing a shepra parameter optimization for option: {option} with settings: {pformat(settings)}")

        num = settings["max_searches"]
        if settings["algorithm"] == "grid":
            algorithm = sherpa.algorithms.GridSearch(num_grid_points=num)
        elif settings["algorithm"] == "random":
            algorithm = sherpa.algorithms.RandomSearch(max_num_trials=num)
        else:
            raise ValueError("algoritthm setting must be one of 'grid', 'random'.")

        parameters = [sherpa.Discrete("batch_size", [settings["batch_from"], settings["batch_to"]])]
        study = sherpa.Study(parameters=parameters, algorithm=algorithm, lower_is_better=True, disable_dashboard=False)

        for trial in study:
            batch_size = trial.parameters["batch_size"]
            context = dict()
            start = datetime.now()
            
            try:
                if option == "sloberta_batch":
                    _, _, sloberta_out_num_tokens = self.run_sloberta_batched_pipeline(batch_size)
                    context["sloberta_out_num_tokens"] = sloberta_out_num_tokens
                elif option == "classla_batch":
                    classla_out = self.run_classla_batched_pipeline(batch_size, *copy.deepcopy(args))
                    context["classla_out_len"] = len(classla_out)
                else:
                    raise ValueError(f"Unknown option: {option}")
                duration_secs = (datetime.now()-start).total_seconds()
                study.add_observation(
                    trial=trial, objective=duration_secs, # iteration=i
                    context={"batch_size": batch_size, "optimization_type": option, **context}
                )
            except:
                continue
            finally:
                study.finalize(trial=trial)
                print("Best trial so far:", study.get_best_result())

        print(f"Finished study for option '{option}': {study.get_best_result()}")



    def process_classla_replacement_words(self, classla_out, sloberta_repl_word_probabilities_list, multi_pass=False):
        for sent in self.sentences:
            if multi_pass == True:
                masked_senteces = sent.masked_sentece_inx_list_multi_pass
            else:
                masked_senteces = sent.masked_sentece_inx_list
            
            for word_masked_inx in masked_senteces:
                if multi_pass == True:
                    # setting new word replacements. first clear the list
                    sent.words[word_masked_inx].clear_word_replacements_multi_pass()

                for _ in range(self.sloberta_kwargs["topk"]):
                    word_replacements = classla_out.pop(0)  # a list of classla words (=sentence)
                    word_replacement = word_replacements[word_masked_inx]  # filter out only the current mask sloberta word replacement

                    assert word_replacement.id == sent.words[word_masked_inx].id, \
                        f"Word ids should be the same.\nSent:{sent.text}\nOrig: {sent.words[word_masked_inx]}\nRepl:{word_replacement}"

                    sloberta_certainty = sloberta_repl_word_probabilities_list.pop(0)
                    sent.words[word_masked_inx].add_word_replacement(word_replacement, sloberta_certainty=sloberta_certainty, multi_pass=multi_pass)  # set sloberta certainty TODO: read from on sloberta

                # all sloberta word replacements set - now use them to calculate case, number prediction
                sent.words[word_masked_inx].set_word_form_predictions(skip_unclassified=self.skip_unclassified, multi_pass=multi_pass)


    def calculate_word_form_predictions_from_suggested_word_replacements(self, thresholds: Thresholds, multi_pass=False):
        # NOTE: self.process_classla_replacement_words should be called before this func
        for sent in self.sentences:
            if multi_pass == True:
                masked_sentences = sent.masked_sentece_inx_list_multi_pass
            else:
                masked_sentences = sent.masked_sentece_inx_list

            # multi_pass_continue
            for word_masked_inx in masked_sentences:
                # reset previously set word_masked_inx form predictions
                sent.words[word_masked_inx].reset_word_form_predictions()
                # all replacement words for current orig word are now set -> calculate prediction
                sent.words[word_masked_inx].calculate_word_form_predictions_from_suggested_word_replacements(thresholds, multi_pass=multi_pass)

    def set_predictions(self, thresholds: Thresholds, multiple_passes=True, print_verbose_all_words=False) -> float:
        self.calculate_word_form_predictions_from_suggested_word_replacements(thresholds, multi_pass=False)

        multi_pass_continue = multiple_passes
        if self.verbose and multi_pass_continue:
            print("Using multiple model passes option.")


        COUNT_MULTI_PASS_MAX = 20
        count_multi_pass = 0
        debug_dt_multi_pass = datetime.now()
        # run now corrected essay through the main pipeline again, to see if there
        # is any other thing to correct - loop untill there is none
        while multi_pass_continue:

            for sent in self.sentences:
                sent.set_masked_sentences_list_multi_pass()
                sent.multi_pass_continue = False


            sloberta_out, sloberta_repl_word_probabilities_list, sloberta_out_len = toma.simple.batch(self.run_sloberta_batched_pipeline, self.tokens_batch_max_sloberta, multi_pass=True, tqdm_enable=False)
            classla_out = toma.simple.batch(self.run_classla_batched_pipeline, self.tokens_batch_max_classla, sloberta_out, sloberta_out_len, multi_pass=True, tqdm_enable=False)
            self.process_classla_replacement_words(classla_out, sloberta_repl_word_probabilities_list, multi_pass=True)
            self.calculate_word_form_predictions_from_suggested_word_replacements(thresholds, multi_pass=True)

            for sent in self.sentences:
                for w_inx in sent.masked_sentece_inx_list_multi_pass:
                    if sent.words[w_inx].eval_has_new_replacement_w:
                        # at least one word has new replacement
                        # we should process this sentence again
                        sent.multi_pass_continue = True
                        continue

            if self.verbose:
                logger.info(f"{str(yellow(bold('Multi passes: (green original)|(red wrong)|(yellow corrected) words:')))}\n{self.text_orig_and_eval_corrected_colored}\n ------------ ")

            count_multi_pass += 1
            if all([not s.multi_pass_continue for s in self.sentences]) or count_multi_pass >= COUNT_MULTI_PASS_MAX:
                # break the loop
                multi_pass_continue = False


        if self.verbose and multiple_passes:
            logger.info(f"Multiple pipeline passes took: {datetime.now()-debug_dt_multi_pass}")


        if print_verbose_all_words:
            logger.info(f"{str(bold(yellow('After last pipeline pass: (red original)|(green corrected) words:')))}\n{self.text_orig_and_corrected_colored}")
            for s in self.sentences:
                for w in s.words:
                    print(w.verbose_string())

        if self.verbose:
            if self.eval_mode:
                print(str(yellow(bold('Last essay pass: (green original)|(red wrong)|(yellow corrected) words:'))))
            else:
                print(str(yellow(bold('Last essay pass: (red original)|(green corrected) words:'))))
            print(self.text_orig_and_eval_corrected_colored)



    def eval_calculate_word_form_predictions_from_suggested_word_replacements(self, thresholds: Thresholds=None, verbose=False, multiple_passes=True, print_conf_metrices=False) -> float:
        thresholds = thresholds if thresholds else self.thresholds
        
        for sent in range(len(self.sentences)):
            for word in range(len(self.sentences[sent].words)):
                # setting new word replacements. first clear the list
                self.sentences[sent].words[word].reset_word_form_predictions()
                self.sentences[sent].words[word].reset_word_form_predictions_cn(multi_pass=True)
                self.sentences[sent].words[word].clear_word_replacements_multi_pass()
                self.sentences[sent].reset_multi_pass_continue()
        
        self.set_predictions(thresholds, multiple_passes=multiple_passes)

        # calculate correction_percentage to evaluate how good the thresholds are set
        num_sentences, num_wrong_words, num_correct_replacements, conf_matrix, num_all_words = self.eval_analisys(verbose=verbose)
            
        try:
            tp, fp, fn, tn = conf_matrix.tp, conf_matrix.fp, conf_matrix.fn, conf_matrix.tn
            accuracy = (tp+tn)/(tp+fp+fn+tn)
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            f1_score = tp/(tp+(0.5*(fp+fn)))
        except Exception as e:
            tp, fp, fn, tn = -1, -1, -1, -1
            accuracy = -1
            precision = -1
            recall = -1
            f1_score = -1
            print("ERROR calculating eval metrices.", e)
            
        return num_sentences, num_wrong_words, conf_matrix, accuracy, precision, recall, f1_score, num_all_words


    def run_sloberta_batched_pipeline(self, tokens_batch_max_sloberta, multi_pass=False, tqdm_enable=True):
        sloberta_topk = self.sloberta_kwargs["topk"]

        t_bar = tqdm(
            total=len(self.sentences),
            desc=f"Processing sloberta batches of max token size {tokens_batch_max_sloberta}. On sentence",
            disable=not tqdm_enable)
        sloberta_out = []
        s_progress_prev = 0
        for sentences_batch, s_progress in self.get_batch_generator_for_masked_sentences_sloberta(tokens_batch_max=tokens_batch_max_sloberta, multi_pass=multi_pass):
            batch_pretokenized_sloberta, num_tokens_in_batch = self.pretokenize_sloberta_batch(sentences_batch, tokens_batch_max_sloberta)
            # logger.debug(f"Made a masked sentences sloberta batch with: {num_tokens_in_batch} number of tokens")  #. Sentences:\n{pformat(input_batch)}")
            
            processed_sloberta = self.nlp_sloberta(batch_pretokenized_sloberta, **self.sloberta_kwargs) # TODO change params
            assert all([len(i) == sloberta_topk for i in processed_sloberta]), \
                f"Not all sloberta out sentences have {sloberta_topk} word predictions. processed_sloberta:\n{pformat(processed_sloberta)}"
            sloberta_out += processed_sloberta
            
            # TODO: test if really needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t_bar.update(s_progress - s_progress_prev)
            t_bar.set_postfix({'num_tokens_in_batch': num_tokens_in_batch})
            s_progress_prev = s_progress
            
        t_bar.close()

        # converts list of lists (of topk sloberta replacement words) to one big list
        repl_words_probabilities_list_list = [[w["score"] for w in words] for words in sloberta_out]
        sloberta_repl_word_probabilities_list = list(itertools.chain.from_iterable(repl_words_probabilities_list_list))

        sloberta_out_num_tokens = sum([len(i) for i in sloberta_out])

        return sloberta_out, sloberta_repl_word_probabilities_list, sloberta_out_num_tokens

    def run_classla_batched_pipeline(self, tokens_batch_max_classla, sloberta_out, sloberta_out_len, multi_pass=False, tqdm_enable=True):
        t_bar = tqdm(
            total=len(self.sentences),
            desc=f"Processing classla batches of max token size {tokens_batch_max_classla}. On sentence",
            disable=not tqdm_enable)
        classla_out =  []
        s_progress_prev = 0
        for sentences_batch, s_progress in self.get_batch_generator_for_masked_sentences_classla(sloberta_out, tokens_batch_max_classla, multi_pass=multi_pass):
            num_tokens_in_batch = sum([len(i) for i in sentences_batch])
            # logger.debug(f"Made a classla batch with: {num_tokens_in_batch} number of tokens")  #. Sentences:\n{pformat(input_batch)}")
            
            processed_classla = self.cl_pipeline_pretok(sentences_batch)
            processed_classla = [sent.words for sent in processed_classla.sentences]
            classla_out += processed_classla  # TODO: could also only append the words. no need for whole sentences

            # TODO: check if really needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t_bar.update(s_progress - s_progress_prev)
            t_bar.set_postfix({'num_tokens_in_batch': num_tokens_in_batch})
            s_progress_prev = s_progress

        assert len(classla_out) == sloberta_out_len, \
            f"The number of sentences classla_out ({len(classla_out)}) and SlobertaMSD output ({sloberta_out_len}) should be the same."
            
        return classla_out

    def get_batch_generator_for_masked_sentences_sloberta(self, tokens_batch_max,  multi_pass=False) -> Generator[List[str], None, None]:
        def get_num_tokens_sloberta_for_sent(sent):
            tokens = self.tokenizer_sloberta(sent, add_special_tokens=True).get("input_ids")
            assert tokens and isinstance(tokens, list) and len(tokens) > 1, f"tokenizer out is weird: {tokens}"
            return len(tokens)
        
        sentences_batch = []
        curr_batch_num_sentences = 0
        curr_max_tokens = -1  # current max number of tokens in the sentence
        for s_progress, s in enumerate(self.sentences):
            if multi_pass == True:
                sent_masked_list = s.masked_sentece_str_list_multi_pass
            else:       
                sent_masked_list = s.masked_sentece_str_list  # ['<mask> je samo dokaz. ', 'Želela <mask> samo dokaz. ', 'Želela je samo <mask>. ']

            for sent_masked in sent_masked_list:
                num_tokens = get_num_tokens_sloberta_for_sent(sent_masked)
                assert num_tokens <= self.tokenizer_sloberta.model_max_length, \
                    f"Sloberta num tokens {num_tokens} in sentence should be <= {self.tokenizer_sloberta.model_max_length}. Sentence:\n{sent_masked}"
                curr_max_tokens = num_tokens if num_tokens > curr_max_tokens else curr_max_tokens
                curr_batch_num_sentences += 1
                assert curr_max_tokens > 0

                batch_tokens_next = curr_batch_num_sentences * curr_max_tokens
                if batch_tokens_next <= tokens_batch_max:
                    sentences_batch.append(sent_masked)

                elif sentences_batch:
                    yield copy.deepcopy(sentences_batch), s_progress
                    sentences_batch = [sent_masked]
                    curr_batch_num_sentences = 1
                    curr_max_tokens = num_tokens
                else:
                    raise ValueError(f"It seems that the number of tokens in first sentence is bigger than tokens_batch_max. "
                                     f"{batch_tokens_next} </= {tokens_batch_max}. Try to increase the max number of tokens per batch.")
        
        if sentences_batch:
            yield sentences_batch, len(self.sentences)

    def pretokenize_sloberta_batch(self, input_batch, tokens_batch_max) -> BatchEncoding:
        inputs_pretokenized = self.tokenizer_sloberta(
            input_batch,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation="do_not_truncate",
        )

        num_tokens_in_batch = sum([len(i) for i in inputs_pretokenized.get("input_ids")])
        assert num_tokens_in_batch <= tokens_batch_max, f"There should be less tokens than {tokens_batch_max} per batch, is: {num_tokens_in_batch}\n{inputs_pretokenized}"
        return inputs_pretokenized, num_tokens_in_batch

    def get_batch_generator_for_masked_sentences_classla(self, sloberta_out, tokens_batch_max, multi_pass=False) -> Generator[List[str], None, None]:
        sentences_batch = []
        curr_batch_num_tokens = 0
        for s_inx in range(len(self.sentences)):
            if multi_pass == True:
                masked_sentences = self.sentences[s_inx].masked_sentece_inx_list_multi_pass
            else:
                masked_sentences = self.sentences[s_inx].masked_sentece_inx_list
                
            for w_inx in masked_sentences:
                word_replacements = sloberta_out.pop(0)  # this list is empty at the end!
                for w in word_replacements:
                    fake_sentence = self.sentences[s_inx].get_sentence_text_list_with_replaced_ith_word(w_inx, w.get("token_str"))
                    sent_len = len(fake_sentence)

                    batch_tokens_next = curr_batch_num_tokens + sent_len
                    if batch_tokens_next <= tokens_batch_max:
                        sentences_batch.append(fake_sentence)
                        curr_batch_num_tokens += sent_len
                    elif sentences_batch:
                        yield copy.deepcopy(sentences_batch), s_inx
                        sentences_batch = [fake_sentence]
                        curr_batch_num_tokens = sent_len
                    else:
                        raise ValueError(f"It seems that the number of tokens in first sentence is bigger than tokens_batch_max. "
                                        f"{batch_tokens_next} </= {tokens_batch_max}. Try to increase the max number of tokens per batch.")

        if sentences_batch:
            yield sentences_batch, len(self.sentences)


    @ property
    def text(self):
        string = ""
        for s in self.sentences:
            string += "\n" if s.is_start_of_new_paragraph else ""
            string += f"{s.text}"
        return str(string).strip()

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.__str__()

    def verbose_print(self):
        string = ""
        for s in self.sentences:
            string += "##### NEW PARAGRAPH #####\n\n" if s.is_start_of_new_paragraph else ""
            string += f"{s.verbose_print()}\n"
        return str(string)


    @ property
    def text_orig_colored(self):
        string = ""
        for s in self.sentences:
            string += f"{s.text_orig_colored}"
        return str(string)

    @ property
    def text_corrected(self):
        string = ""
        for s in self.sentences:
            string += f"{s.text_corrected}"
        return str(string)

    @ property
    def text_corrected_colored(self):
        string = ""
        for s in self.sentences:
            string += f"{s.text_corrected_colored}"
        return str(string)

    @ property
    def text_orig_and_corrected_colored(self):
        string = ""
        for s in self.sentences:
            string += f"{s.text_orig_and_corrected_colored}"
        return str(string)

    @ property
    def text_orig_and_eval_corrected_colored(self):
        string = ""
        for s in self.sentences:
            string += f"{s.text_eval_orig_corrected_colored if self.eval_mode else s.text_orig_and_corrected_colored}"
        return str(string)

    def print_corrected_essay(self, only_corrected=True):
        logger.info(f"{str(red('Original input text with (red original)|(green corrected) words:'))}\n{self.text_orig_and_corrected_colored}")
        for s in self.sentences:
            for w in s.words:
                if only_corrected:
                    if w.word_is_corrected():
                        print(w.verbose_string())
                else:
                    print(w.verbose_string())
                    

    @property
    def eval_mode(self):
        return self.eval_params is not None

    @property
    def num_sentences(self):
        return len(self.sentences)

    def __len__(self):
        return self.num_sentences
    
    def __iter__(self):
        for s in self.sentences:
            yield s

    def debug_print_masked_sentences(self):
        logger.info("Generated masked sentences for sloberta:")
        for s in self.sentences:
            if s.masked_sentece_str_list:
                print(s.masked_sentece_str_list)
