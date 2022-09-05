from typing import Union
from classla.models.common.doc import Word as WordClassla
from curtsies.fmtfuncs import green, bold, yellow, red

from DictHelper import DictHelper
from helpers import Thresholds, WordFormChangeCase, WordFormChangeNumber

import logging
logger = logging.getLogger("__main__." + __name__)


class WordMsd:
    """
    Wrapper class, that provides additional MSD tags and other attributes for words.
    """

    def __init__(self, word: WordClassla, dh: DictHelper, sloberta_certainty=0.0):
        assert isinstance(word, WordClassla), f"The word must be of type WordClassla, is: {type(word)}"
        assert isinstance(sloberta_certainty, float) and 0 <= sloberta_certainty <= 1, \
            f"sloberta_certainty bust be a float between 0 and 1, is {sloberta_certainty}"
        
        self._word_classla: WordClassla = word
        self.dh: DictHelper = dh
        self.sloberta_certainty = sloberta_certainty

        self.msd_en = word.xpos
        self.lemma = word.lemma
        case, gender, number, owner_number, _, self.msd_sl, _ = self.dh.get_msd_info(word.xpos)
        # self.msd_sl = self.dh.convert_msd_en_to_sl(self.msd_en)
        self.case: int = case
        self.gender: str = gender
        self.number: int = number
        self.owner_number: int = owner_number
        # print(vars(self))

        self.space_after = False if word.misc and "SpaceAfter=No" in word.misc else True
        self.id = self._word_classla.id

    @ property
    def text(self) -> str:
        return str(self._word_classla.text)

    def is_to_be_predicted(self, thresholds: Thresholds, eval_wrong_word_case, eval_wrong_word_num):
        """Check if the word is to be masked (and then put into sloberta pipeline for predictions)."""

        assert isinstance(thresholds, Thresholds), "thresholds arg should be of type Thresholds"
        
        case = eval_wrong_word_case if eval_wrong_word_case != -1 else self.case
        num = eval_wrong_word_num if eval_wrong_word_num != -1 else self.number
        
        thresholds_list_cases = [f"case{case}{case_to}" for case_to in range(1, 7, 1)]
        thresholds_list_nums  = [f"number{num}{num_to}" for num_to in range(1, 4, 1)]
        thresholds_list = thresholds_list_cases + thresholds_list_nums
        for threshold_name in thresholds_list:
            if hasattr(thresholds, threshold_name):
                threshold_percentage = getattr(thresholds, threshold_name)
                if threshold_percentage < 1:
                    # print("#####", self.text, threshold_percentage, threshold_name, thresholds_list)
                    return True
        
        return False

    @property
    def has_case(self):
        return self.case in {1, 2, 3, 4, 5, 6}

    @property
    def has_number(self):
        return self.number in {1, 2, 3}
    
    def eval_get_new_word_form(self, form_change: Union[WordFormChangeCase, WordFormChangeNumber]) -> str:
        assert isinstance(form_change, WordFormChangeCase) or isinstance(form_change, WordFormChangeNumber)
        
        case_wanted, number_wanted = -1, -1
        if isinstance(form_change, WordFormChangeCase) and self.case == form_change.case_from:
            case_wanted = form_change.case_to
        if isinstance(form_change, WordFormChangeNumber) and self.number == form_change.number_from:
            number_wanted = form_change.number_to

        # logger.info(f"number_wanted: {number_wanted}, case_wanted: {case_wanted}")
        if case_wanted != -1 or number_wanted != -1:
            word_fake_replacement = self.dh.change_word_form_by_msd(
                                            self.text,
                                            self.msd_sl,
                                            case_wanted=case_wanted,
                                            number_wanted=number_wanted
                                        )
            if word_fake_replacement and word_fake_replacement.word_text != self.text:
                # repl word is different than origial -> ok
                # get new fake word case and number from msd tag
                return word_fake_replacement.word_text, case_wanted, number_wanted
        return None, None, None



    def __str__(self) -> str:
        string = f"{str(green(self.text)): <24} "
        string += f"{self.lemma: <14} "
        string += f"c:{self.case: 2d}  "
        string += f"g:{self.gender: <1}  "
        string += f"n:{self.number: 1d}  "
        # string += f"own_n: {self.owner_number: 1d} "
        if self.sloberta_certainty > 0:
            string += f"{float(self.sloberta_certainty)*100: 10.3f}% "
        return str(string)

    def __repr__(self) -> str:
        return self.__str__()
