from classla.models.common.doc import Sentence as SentenceClassla
from curtsies.fmtfuncs import green, bold, yellow, red
from typing import Dict, Iterable, List, Tuple, Union
from DictHelper import DictHelper
from constants import *
from WordHelper import WordHelper
from WordMsd import WordMsd
import random
import re
import numpy as np
from helpers import *

import logging
logger = logging.getLogger("__main__." + __name__)


re_sent_id = re.compile(r"# sent_id = (\d+(?:\.\d+)?)")

re_match_slo_words = re.compile(RE_DEFAULT_TOKEN_STR_FILTER)


class SentenceHelper:
    def __init__(self, cl_sentence: SentenceClassla, dh: DictHelper, eval_params: EvalParams, thresholds: Thresholds):
        self.cl_sentence_orig: SentenceClassla = cl_sentence
        self.dh: DictHelper = dh
        self.eval_params: EvalParams = eval_params
        self.thresholds: Thresholds = thresholds

        self.words: List[WordHelper] = []
        self.is_start_of_new_paragraph = False

        self.func_vec_mask_indeces = np.vectorize(lambda w: w.is_to_be_predicted) # w.eval_is_to_be_predicted if self.eval_mode else w.is_to_be_predicted)
        self.sent_id_classla = None
        # self.func_vec_mask_indeces = np.vectorize(lambda w_inx: w.is_to_be_predicted)

        self.masked_sentece_str_list = []
        self.masked_sentece_inx_list = []
        
        self.masked_sentece_str_list_multi_pass = []
        self.masked_sentece_inx_list_multi_pass = []
        self.masked_sentece_str_list_multi_pass_old = []
        self.masked_sentece_inx_list_multi_pass_old = []
        
        self.multi_pass_continue = True
        
        self.eval_predictions_set = False

        self.process_sentence()

    def compare_sentences(self, sent):
        if sent.text == self.cl_sentence_orig.text \
            and self.masked_sentece_inx_list == sent.masked_sentece_inx_list:
                return True
        return False

    def process_sentence(self):
        misc_classla = self.cl_sentence_orig.to_dict()[1]
        self.sent_id_classla, self.is_start_of_new_paragraph = self._extract_from_misc(misc_classla)

        # create WordHelper objs and append them to list
        for w in self.cl_sentence_orig.words:
            word = WordHelper(w, self.dh, self.thresholds)
            self.words.append(word)

        if self.eval_mode:
            # metrics = self._get_words_metrics()
            # logger.info(f"In eval mode. eval_params:\n{self.eval_params}")
            self.set_wrong_words()

        self.masked_sentece_str_list, self.masked_sentece_inx_list = self._get_masked_sentences_list()

    def eval_iter_predicted_replacements(self) -> Iterable[WordHelper]:
        for w in self.words:
            if w.eval_has_new_replacement_w or w.has_wrong_word:
                yield w

    def set_wrong_words(self) -> bool:
        replacements = list()
        for i, w in enumerate(self.words):
            word: WordMsd = w.word_orig
            wrong_words = []
            for eval_param in self.iterate_eval_word_form_changes():
                wrong_word, wrong_w_case, wrong_w_num = word.eval_get_new_word_form(eval_param)
                if wrong_word:
                    wrong_words.append((wrong_word, wrong_w_case, wrong_w_num))
            if len(wrong_words) > 0:
                wrong_word, wrong_w_case, wrong_w_num = random.sample(wrong_words, 1)[0]
                replacements.append(
                    (WrongWord(inx_words=i, word_text=wrong_word), wrong_w_case, wrong_w_num)
                )
                
            
        # select only a subset of wrong words and set them in the sentence
        if 0 < self.eval_params.num_wrong_words <= len(replacements):
            wrong_words: List[WrongWord] = random.sample(replacements, self.eval_params.num_wrong_words)
            for w, wrong_w_case, wrong_w_num in wrong_words:
                # print("SETTING WRONG WORD:", w.word_text," to index", w.inx_words)
                self.words[w.inx_words].set_wrong_word(w.word_text, wrong_w_case, wrong_w_num)
                # logger.debug(f"{self.words[w.inx_words].text} -> {self.words[w.inx_words].eval_wrong_word}")
            assert self.num_wrong_words == self.eval_params.num_wrong_words
            logger.debug(f"Created wrong word sentence: {self.text_orig_and_wrong_colored}")
            return True

        return False

    def _eval_check_if_predictions_set(self):
        assert self.eval_mode
        assert self.eval_params.num_wrong_words > 0, f"There should be at least 1 wrong word set in eval params: {self.eval_params}"
        
        # if sum([w.eval_is_to_be_predicted for w in self.words]) > 0:
        #     self.eval_predictions_set = True
        
        num_predictions_wanted = sum([w.eval_is_to_be_predicted for w in self.words])
        # num_predictions = sum([w.eval_has_new_replacement_w for w in self.words])
        if num_predictions_wanted == self.eval_params.num_wrong_words > 0:
        # if num_predictions_wanted > 0 and not self.eval_params.num_wrong_words == num_predictions_wanted == num_predictions:
        #     logger.debug(f"Resetting predictions for sentence:\n{self.text_orig_and_wrong_colored}")
        #     self.reset_predictions_and_wrong_words()
        #     logger.debug(f"Reset done:\n{self.text_orig_and_wrong_colored}")
        # elif self.eval_params.num_wrong_words == num_predictions_wanted == num_predictions > 0:
            self.eval_predictions_set = True

    def reset_predictions_and_wrong_words(self):
        for w in self.words:
            w.reset_wrong_word()
            w.reset_selected_str()

    def iterate_eval_word_form_changes(self):
        
        for case_from, case_to in self.eval_params.case_from_to:
            yield WordFormChangeCase(case_from=case_from, case_to=case_to)

        for number_from, number_to in self.eval_params.number_from_to:
            yield WordFormChangeNumber(number_from=number_from, number_to=number_to)

    def _get_masked_sentences_list(self, multi_pass=False):
        """
        returns a list of masked sentences like so: ['<mask> je samo dokaz. ', 'Želela <mask> samo dokaz. ', 'Želela je samo <mask>. ']
        """
        assert self.words and isinstance(self.words, list) and all([isinstance(w, WordHelper) for w in self.words]), "self.words must be loaded first"

        masked_sentece_str_list = []
        masked_sentece_inx_list = []
        words_ideces = self.get_mask_bools_list(self.words, multi_pass=multi_pass)
        
        if multi_pass and not self.multi_pass_continue:
            return [],[]
        
        for w_inx, w_mask_bool in enumerate(words_ideces):
            if w_mask_bool:
                masked_sentece = self.get_masked_sentence(w_inx, multi_pass=multi_pass)
                masked_sentece_str_list.append(masked_sentece)
                masked_sentece_inx_list.append(w_inx)

        return masked_sentece_str_list, masked_sentece_inx_list

    def set_masked_sentences_list_multi_pass(self):
        self.masked_sentece_str_list_multi_pass, \
            self.masked_sentece_inx_list_multi_pass \
                = self._get_masked_sentences_list(multi_pass=True)
    
    def reset_multi_pass_continue(self):
        self.multi_pass_continue = True

    def set_old_masked_sentences_list_multi_pass(self):
        self.masked_sentece_str_list_multi_pass_old = self.masked_sentece_str_list_multi_pass
        self.masked_sentece_inx_list_multi_pass_old = self.masked_sentece_inx_list_multi_pass

    def _extract_from_misc(self, misc_classla):
        sent_id_classla = re_sent_id.search(misc_classla)
        sent_id_classla = sent_id_classla.group(1) if sent_id_classla and sent_id_classla.group(1) else None
        assert sent_id_classla, f"Classla seems to not have provided the sentence id in misc:\n{misc_classla}"

        is_start_of_new_paragraph = True if "newpar" in misc_classla else False
        return sent_id_classla, is_start_of_new_paragraph

    def get_mask_bools_list(self, words, multi_pass):
        if multi_pass == True:
            return np.vectorize(lambda w: w.is_to_be_predicted_multi_pass)(words)

        return self.func_vec_mask_indeces(words)

    def get_masked_sentence(self, w_inx_to_mask, multi_pass):
        sent = ""
        for w_inx, w in enumerate(self.words):
            if w_inx_to_mask == w_inx:
                sent += SLOBERTA_MCD_MASK_TOKEN_STR
            else:
                if multi_pass == True:
                    sent += w.word_selected_str if w.eval_has_new_replacement_w else w.text_wrong if w.has_wrong_word else w.text
                else:
                    sent += w.text_wrong if w.has_wrong_word and self.eval_mode else w.text
            sent += " " if w.space_after else ""
        return sent

    def get_sentence_with_replaced_ith_word(self, w_inx_to_replace, word_replacement) -> str:
        sent = ""
        for w_inx, w in enumerate(self.words):
            sent += word_replacement if w_inx_to_replace == w_inx else w.text
            sent += " " if w.space_after else ""
        return sent

    def get_sentence_text_list_with_replaced_ith_word(self, w_inx_to_replace, word_replacement) -> List[str]:
        sent = []
        for w_inx, w in enumerate(self.words):
            sent.append(word_replacement if w_inx_to_replace == w_inx else w.text)
            # sent += " " if w.space_after else ""
        return sent

    def get_masked_sentences_list_list(self):
        assert len(self.masked_sentece_str_list) == len(self.masked_sentece_inx_list)
        
        sentences = []
        for w_inx in self.masked_sentece_inx_list:
            for w_repl in self.words[w_inx].word_replacements:
                sentence_list = self.get_sentence_text_list_with_replaced_ith_word(w_inx, w_repl.text)
                sentences.append(sentence_list)
        
        return sentences

    @ property
    def text(self):
        sent = ""
        for w in self.words:
            sent += w.word_selected_str if w.word_selected_str else w.text
            sent += " " if w.space_after else ""
        return sent

    def verbose_print(self) -> str:
        string = ""
        for w in self.words:
            string += f"{w.verbose_string()}\n"
        return str(string)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.__str__()

    @ property
    def text_orig_colored(self):
        sent = ""
        for w in self.words:
            sent += f"{str(yellow(w.text)) if w.has_new_replacement_w else w.text}"
            sent += " " if w.space_after else ""
        return sent

    @ property
    def text_corrected(self):
        sent = ""
        for w in self.words:
            sent += w.word_selected_str if w.eval_has_new_replacement_w else w.text_wrong if w.has_wrong_word else w.text
            sent += " " if w.space_after else ""
        return sent

    @ property
    def text_corrected_colored(self):
        sent = ""
        for w in self.words:
            sent += f"{str(green(w.word_selected_str)) if w.eval_has_new_replacement_w else w.text}"
            sent += " " if w.space_after else ""
        return sent

    @ property
    def text_orig_and_corrected_colored(self):
        sent = ""
        for w in self.words:
            sent += f"{str(red(w.text))}|{str(green(bold(w.word_selected_str)))}" if w.eval_has_new_replacement_w and w.text != w.word_selected_str else w.text
            sent += " " if w.space_after else ""
        return sent

    @ property
    def text_eval_orig_corrected_colored(self):
        assert self.eval_mode, f"'{self.text_eval_orig_corrected_colored.__name__}'should not be called if not in eval mode."
        sent = ""
        for w in self.words:
            sent += f"{str(green(w.text))}|{str(red(w.text_wrong))}" if w.has_wrong_word else w.text
            sent += f"|{str(bold(yellow(w.word_selected_str)))}" if w.eval_has_new_replacement_w else ""
            sent += " " if w.space_after else ""
        return sent

    @ property
    def text_orig_and_wrong_colored(self):
        sent = ""
        for w in self.words:
            sent += f"{str(green(w.text))}|{str(red(w.text_wrong))}" if w.has_wrong_word else w.text
            sent += " " if w.space_after else ""
        return sent

    @property
    def eval_mode(self):
        return self.eval_params is not None

    @property
    def num_wrong_words(self):
        return sum([w.has_wrong_word for w in self.words])