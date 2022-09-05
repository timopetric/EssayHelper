from classla.models.common.doc import Word as WordClassla
from curtsies.fmtfuncs import green, bold, yellow, red
from DictHelper import DictHelper
from WordMsd import WordMsd
from typing import Dict, List, Tuple, Union
import operator
from collections import namedtuple
from helpers import Thresholds, WordForm

import logging
logger = logging.getLogger("__main__." + __name__)

PredictionTuple = namedtuple("PredictionTuple", ["prediction", "percentage"])

class WordHelper:
    dh: DictHelper = None

    def __init__(self, word: WordClassla, dh: DictHelper, thresholds: Thresholds) -> None:
        # original word, as it appeared in the input text
        self.dh: DictHelper = dh
        self.thresholds = thresholds
        self.word_orig: WordMsd = WordMsd(word, self.dh)
        self.word_replacements: List[WordMsd] = []  # sloberta predicted replacements
        self.word_replacements_orig: List[WordMsd] = []  # sloberta predicted replacements

        self.form_predictions_calculated = False
        self.case_prediction: PredictionTuple = None
        self.number_prediction: PredictionTuple = None
        self.gender_prediction: PredictionTuple = None
        self.case_percentages = []
        self.number_percentages = []

        self.form_predictions_calculated_orig = False
        self.case_prediction_orig: PredictionTuple = None
        self.number_prediction_orig: PredictionTuple = None
        self.gender_prediction_orig: PredictionTuple = None
        self.case_percentages_orig = []
        self.number_percentages_orig = []

        # suggested word replacement
        self.word_selected_str = ""

        # wrong word that is set and passed to sloberta in eval mode
        self.eval_wrong_word = ""
        self.eval_wrong_word_case = -1
        self.eval_wrong_word_num = -1

        self.first_sugg_was_below_sloberta_threshold_skipped = False
        self.first_sloberta_sugg_used_for_selection = False

    def set_word_selected_str(self, word: Union[WordMsd, WordForm]):
        assert self.word_selected_str == "", f"It looks like word_selected_str was set before. {self.word_selected_str}"
        assert isinstance(word, WordMsd) or isinstance(word, WordForm)
        word_text = None
        if isinstance(word, WordForm):
            word_text = word.word_text
        elif isinstance(word, WordMsd):
            word_text = word.text
        self.word_selected_str = word_text    

    def add_word_replacement(self, word: WordClassla, sloberta_certainty: float, multi_pass=False):
        w = WordMsd(word, self.dh, sloberta_certainty=sloberta_certainty)
        if multi_pass == True:
            self.word_replacements.append(w)
        else:
            self.word_replacements_orig.append(w)
        
    def clear_word_replacements_multi_pass(self):
        self.word_replacements = []
        # self.word_replacements_orig = []

    def get_word_form_predictions(self, skip_unclassified=True, multi_pass=False):
        """Based on all the WordMsd sloberta suggestions in self.word_replacements
           predict which form the original word should be in
        Returns:
            case_prediction - List[float]: List of sums of percentages sloberta predictions for case
            number_prediction - List[float]: List of sums of percentages sloberta predictions for number
            gender_prediction - List[float]: List of sums of percentages sloberta predictions for gender
        """

        # TODO: use numpy and vectorize -> possible speedup

        case_percentages = [0 for _ in range(7)]
        number_percentages = [0 for _ in range(4)]
        gender_percentages = {"": 0, "m": 0, "f": 0, "n": 0}

        # sum percentages to word form lists
        word_repl_list = self.word_replacements if multi_pass == True else self.word_replacements_orig
        for w in word_repl_list:
            case_percentages[w.case if w.case != -1 else 0] += w.sloberta_certainty
            number_percentages[w.number if w.number != -1 else 0] += w.sloberta_certainty
            gender_percentages[w.gender] += w.sloberta_certainty
        
        # def softmax(x):
        #     l = np.exp(x)/np.sum(np.exp(x),axis=0)
        #     return list(l)
        # case_percentages = softmax(case_percentages)
        # number_percentages = softmax(number_percentages)
        # gender_percentages = softmax(gender_percentages)

        # debug output word form lists
        pprint_case_perc = [f"{float(p)*100: 6.3f}%" for p in case_percentages]
        pprint_number_perc = [f"{float(p)*100: 6.3f}%" for p in number_percentages]
        pprint_gender_perc = [f"{float(gender_percentages[p])*100: 6.3f}%" for p in gender_percentages]
        logger.debug(f"Form predictions for word {str(green(self.word_orig.text))}:")
        logger.debug(f"Case:   {pprint_case_perc}")
        logger.debug(f"Number: {pprint_number_perc}")
        logger.debug(f"Gender: {pprint_gender_perc}")
        
        # find argmax from word form lists
        start_at = 1 if skip_unclassified else 0
        
        case_percentages = case_percentages[start_at:]
        number_percentages = number_percentages[start_at:]
        pred_case_inx =   max(zip(case_percentages, range(len(case_percentages))))[1]
        pred_number_inx = max(zip(number_percentages, range(len(number_percentages))))[1]
        pred_gender = max(gender_percentages.items(), key=operator.itemgetter(1))[0]

        pred_case = -1 if case_percentages[pred_case_inx] < 0.001 else pred_case_inx+start_at
        pred_number = -1 if number_percentages[pred_number_inx] < 0.001 else pred_number_inx+start_at

        # save current word form predictions
        case_prediction = PredictionTuple(pred_case, case_percentages[pred_case_inx])
        number_prediction = PredictionTuple(pred_number, number_percentages[pred_number_inx])
        gender_prediction = PredictionTuple(pred_gender, gender_percentages[pred_gender])

        return case_prediction, number_prediction, gender_prediction, case_percentages, number_percentages

    def set_word_form_predictions(self, skip_unclassified=True, multi_pass=False):
        if multi_pass == True:
            self.case_prediction, self.number_prediction, self.gender_prediction, self.case_percentages, self.number_percentages = self.get_word_form_predictions(skip_unclassified=skip_unclassified, multi_pass=multi_pass)
            self.form_predictions_calculated_orig = True
        else:
            self.case_prediction_orig, self.number_prediction_orig, self.gender_prediction_orig, self.case_percentages_orig, self.number_percentages_orig = self.get_word_form_predictions(skip_unclassified=skip_unclassified, multi_pass=multi_pass)
            self.form_predictions_calculated = True
        # logger.debug(self.pformat_word_predictions())

    def reset_word_form_predictions_cn(self, multi_pass=False):
        if multi_pass == True:
            self.case_prediction = None
            self.number_prediction = None
            self.gender_prediction = None
            self.case_percentages = []
            self.number_percentages  = []
        else:
            self.case_prediction_orig = None
            self.number_prediction_orig = None
            self.gender_prediction_orig = None
            self.case_percentages_orig = []
            self.number_percentages_orig = []

    def reset_word_form_predictions(self):
        self.first_sugg_was_below_sloberta_threshold_skipped = False
        self.first_sloberta_sugg_used_for_selection = False
        
        self.reset_selected_str()
        
    def calculate_word_form_predictions_from_suggested_word_replacements(self, thresholds: Thresholds, multi_pass=False):
        assert isinstance(thresholds, Thresholds), f"Argument {thresholds.__name__} must be of type {Thresholds.__name__}, but is: {type(thresholds)}"
        assert self.word_replacements or self.word_replacements_orig, f"This func should not be called if there are no set word_replacements: {self.word_replacements} or {self.word_replacements_orig}"
        assert not self.word_selected_str, f"This func should not be called if the replacement word was already chosen: '{self.word_selected_str}'"
        assert self.form_predictions_calculated == True, f"{self.set_word_form_predictions.__name__} should be called before this func."

        # ALL TOPK SLOBERTA WORDS LOADED
        word_replacements = self.word_replacements if multi_pass == True else self.word_replacements_orig
        for i, w_repl in enumerate(word_replacements):
            assert 0 <= w_repl.sloberta_certainty <= 1,  \
                f"Word replacement should have a sloberta certainty between 0 and 1, has: {w_repl.sloberta_certainty}"

            if i == 0 and w_repl.sloberta_certainty < thresholds.skip_if_first_sugg_below:
                logger.debug(f"Skipping suggestions for orig word '{w_repl.text}', because first sloberta suggestion is below {w_repl.sloberta_certainty*100:.1f} <= {thresholds.skip_if_first_sugg_below*100:.1f} threshold.")
                self.first_sugg_was_below_sloberta_threshold_skipped = True
                return

            if w_repl.sloberta_certainty >= thresholds.direct_replacement_if_same_lemma and w_repl.lemma == self.lemma:
                logger.debug(f"Directly setting word replacement '{w_repl.text}' for orig word '{self.text}', because sloberta is more than {w_repl.sloberta_certainty*100:.1f}% >= {thresholds.direct_replacement_if_same_lemma*100:.1f} certain and lemma is the same.")
                # TODO: word 's' many times wrongly converts to z -> only look at lemma?
                self.set_word_selected_str(w_repl)
                self.first_sloberta_sugg_used_for_selection = True
                return

        logger.debug(self.pformat_word_predictions(multi_pass)) # log word form predictions

        case_wanted = -1
        case_now = self.eval_wrong_word_case if self.eval_wrong_word_case != -1 else self.word_orig.case
        case_then = self.case_prediction.prediction if multi_pass == True else self.case_prediction_orig.prediction
        if case_now > 0 and case_then > 0 and case_now != case_then:
            threshold_case = getattr(thresholds, f"case{case_now}{case_then}")
            case_percentage = self.case_prediction.percentage if multi_pass == True else self.case_prediction_orig.percentage
            
            if case_percentage >= threshold_case:                
                case_wanted = case_then
            else:
                logger.debug(f"Skipping case prediction for '{self.text}', because it is below threshold:  {case_percentage:3.2f} < {threshold_case:3.2f}")
                    
        number_wanted = -1
        number_now = self.eval_wrong_word_num if self.eval_wrong_word_num != -1 else self.word_orig.number
        number_then = self.number_prediction.prediction if multi_pass == True else self.number_prediction_orig.prediction
        if number_now > 0 and number_then > 0 and number_now != number_then:
            threshold_number = getattr(thresholds, f"number{number_now}{number_then}")
            number_percentage = self.number_prediction.percentage if multi_pass == True else self.number_prediction_orig.percentage
            if number_percentage >= threshold_number:
                number_wanted = number_then
            else:
                logger.debug(f"Skipping number prediction for '{self.text}', because it is below threshold: {number_percentage:3.2f} < {threshold_number:3.2f}")


        if case_wanted != -1 or number_wanted != -1:
            word_orig_text = self.word_orig.text
            word_orig_msd_sl = self.word_orig.msd_sl

            word_new = self.dh.change_word_form_by_msd(word_orig_text, word_orig_msd_sl, case_wanted=case_wanted, number_wanted=number_wanted)
            if word_new:
                logger.debug(f"Predicted word: {str(red(word_new.word_text))}, MSD_sl: {word_new.msd_slo}, MSD_en: {word_new.msd_en}")
                logger.debug(f"Predicted word: {word_new}")
                self.set_word_selected_str(word_new)
            else:
                logger.debug(f"Predicted word correction is not possible for word: {str(red(word_orig_text))}, MSD_sl: {word_orig_msd_sl}.")



    def set_wrong_word(self, text, wrong_w_case, wrong_w_num):
        assert self.text != text, \
            f"Wrong word replacement text should not be the same as original word text. {self.text} == {text}"
        assert isinstance(text, str)
        assert len(text) > 0 
        self.eval_wrong_word_case = wrong_w_case
        self.eval_wrong_word_num = wrong_w_num
        self.eval_wrong_word = text
        
    def reset_wrong_word(self):
        self.eval_wrong_word = ""
        self.eval_wrong_word_case = -1
        self.eval_wrong_word_num = -1

    def reset_selected_str(self):
        self.word_selected_str = ""

    def pformat_word_predictions(self, multi_pass: bool) -> str:
        if not self.form_predictions_calculated:
            logger.warning(f"{self.pformat_word_predictions.__name__} called before new word form predictions were calculated.")
            return ""

        case_percentages = self.case_percentages if multi_pass == True else self.case_percentages_orig
        number_percentages = self.number_percentages if multi_pass == True else self.number_percentages_orig
        case_prediction = self.case_prediction if multi_pass == True else self.case_prediction_orig
        number_prediction = self.number_prediction if multi_pass == True else self.number_prediction_orig
        
        case_percentages_str = ", ".join([
            f"{c*100:4.1f}%"
            for i, c in
            enumerate(case_percentages)
        ])
        number_percentages_str = ", ".join([
            f"{n*100:4.1f}%"
            for i, n in
            enumerate(number_percentages)
        ])
        
        start_words = "New predictions for" if multi_pass == True else "Predictions for"
        return str(f"{start_words} for word {self.text_wrong if self.text_wrong else self.text}:\n"  #original word: '{str(green(self.word_orig.text))}' with MSD_sl tag {self.word_orig.msd_sl}:\n"
                   f"          Case:   {case_prediction.prediction:2d} {case_prediction.percentage*100:6.2f}% cases:   [{case_percentages_str}]\n"
                   f"          Number: {number_prediction.prediction:2d} {number_prediction.percentage*100:6.2f}% numbers: [{number_percentages_str}]\n")
                #    f"      Gender: {self.gender_prediction.prediction:>2s} {self.gender_prediction.percentage*100:6.2f}%")

    @ property
    def id(self):
        return self.word_orig.id

    @property 
    def has_wrong_word(self):
        return self.eval_wrong_word != ""
    
    @property
    def has_correct_repl(self):
        return self.text == self.word_selected_str or self.first_sugg_was_below_sloberta_threshold_skipped or self.first_sloberta_sugg_used_for_selection

    @ property
    def has_new_replacement_w(self):
        return self.word_selected_str != "" and self.word_selected_str != self.text

    @ property
    def eval_has_new_replacement_w(self):
        return self.word_selected_str != ""

    @ property
    def text(self):
        return self.word_orig.text
        # return self.word_selected_str if self.word_selected_str else self.word_orig.text

    @ property
    def text_wrong(self):
        return self.eval_wrong_word

    @ property
    def lemma(self):
        return self.word_orig.lemma

    @ property
    def space_after(self):
        return self.word_orig.space_after

    @ property
    def is_to_be_predicted(self):
        return self.word_orig.is_to_be_predicted(self.thresholds, self.eval_wrong_word_case, self.eval_wrong_word_num)

    @ property
    def is_to_be_predicted_multi_pass(self):
        return self.is_to_be_predicted and not self.word_selected_str

    @ property
    def eval_is_to_be_predicted(self):
        # return self.is_to_be_predicted  # TODO: maybe has unsettling consequences ... does
        return self.has_wrong_word    # this should be ok

    def __str__(self) -> str:
        return self.word_orig.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def selected_word_is_the_same_as_orig(self):
        return self.word_selected_str == self.text

    def word_is_corrected(self):
        return self.word_selected_str and self.word_selected_str != self.text

    def verbose_string(self):
        string = f"Original word:  {self.word_orig}"
        string += " (suggestions were skipped)" if self.first_sugg_was_below_sloberta_threshold_skipped else ""

        if self.has_wrong_word:
            string += f"\n                {str(yellow(self.eval_wrong_word))} c:{self.eval_wrong_word_case} n:{self.eval_wrong_word_num} <-- {str(bold('WRONG WORD'))}"

        if self.word_replacements_orig:
            sel_word_str = str(green(bold(self.word_selected_str))) if self.selected_word_is_the_same_as_orig() else str(red(bold(self.word_selected_str)))
            string += f"\n                {sel_word_str if self.word_selected_str else self.word_orig} <-- {str(bold('SELECTED REPL. WORD'))}"

            string += " (thresholded sloberta sugg used)" if self.first_sloberta_sugg_used_for_selection else ""
            string += "\n" if string[-1] != "\n" else ""
            string += str(bold("    SloBERTa predicted word replacements:\n"))
            for s in self.word_replacements_orig:
                string += f"\t\t{s}\n"
            string += "        "+self.pformat_word_predictions(multi_pass=False) if self.form_predictions_calculated else ""

            if self.word_replacements:
                # string += str(yellow(bold("    Suggested words after multiple passes:\n")))
                string += str(bold(f"    SloBERTa predicted word replacements {yellow('after multiple passes')}:")) +"\n"
                for s in self.word_replacements:
                    string += f"\t\t{s}\n"
                string += "        "+self.pformat_word_predictions(multi_pass=True) if self.form_predictions_calculated else ""

        return str(string)[:-1] if string[-1] == "\n" else str(string)  # remove last new line char
