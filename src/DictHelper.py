from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from pprint import pprint, pformat
from pathlib import Path
from curtsies.fmtfuncs import green, bold, yellow, red
import logging

from helpers import *
from constants import *


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def read_sloleks_to_dict(filepath):
    # https://www.clarin.si/repository/xmlui/handle/11356/1230
    word_dict_forms = dict()
    word_dict_pairs = dict()
    with open(filepath, "r") as f:
        for line in tqdm(f.readlines(), desc="Loading Sloleks"):
            line = re.sub(r"\s+", " ", line)
            line = line.split(" ")

            # print(line)
            # and not isinstance(word_dict_forms[line[1]], dict)
            if line[1] not in word_dict_forms:
                word_dict_forms[line[1]] = dict()
            # else:
                # pprint(word_dict_forms[line[1]])

            # if line[2] in word_dict_forms[line[1]]:
            #     pass
            word_dict_forms[line[1]][line[2]] = line[0]

            # word_dict_pairs[f"{line[0]}-{line[2]}"] = line[1]
            if line[0] not in word_dict_pairs:
                word_dict_pairs[line[0]] = set()

            word_dict_pairs[line[0]].add(line[1])

    return word_dict_forms, word_dict_pairs


def read_msd_xml_to_dict(filepath, language) -> dict:
    """Read slovenian/english msd specification xml file to a python dict object.

    Args:
        filepath (str, optional): path to the xml file with slovenian specifications.

    Returns:
        dict: a dictionary of msd tag, description pairs
    """
    # https://raw.githubusercontent.com/clarinsi/mte-msd/master/xml/msd-sl.spc.xml
    # http://nl.ijs.si/ME/Vault/V5/msd/html/msd-sl.html#msd.msds-sl
    # https://raw.githubusercontent.com/mgrcar/OPA/6f0b45920f562db5a32aa23ac258ab4c759c2a51/Obeliks/LemmatizerTrain/MsdSpecsSloSloCodes.txt

    if language not in ["sl", "en"]:
        raise ValueError("Dict msd language can either be 'sl' or 'en'.")

    soup = None
    with open(filepath, "r") as f:
        soup = BeautifulSoup(f.read(), "xml")

    msd_dict = dict()
    for row in soup.find_all("row", role="msd"):
        cells = row.find_all("cell")
        msd_en = cells[0].text
        msd_sl = cells[2].text
        description = cells[3].text
        examples = cells[6].text
        # print(msd_sl)

        msd_lang = msd_sl if language == "sl" else msd_en
        # msd_dict[msd_en] = {
        msd_dict[msd_lang] = {
            "description": description.strip(),
            "examples": examples.strip(),
            "case": 1 if "sklon=imenovalnik" in description else
                    2 if "sklon=rodilnik" in description else
                    3 if "sklon=dajalnik" in description else
                    4 if "sklon=tožilnik" in description else
                    5 if "sklon=mestnik" in description else
                    6 if "sklon=orodnik" in description else -1,
            "gender": "f" if "spol=ženski" in description else
                      "m" if "spol=moški" in description else
                      "n" if "spol=srednji" in description else "",
            "number": 1 if "število=ednina" in description else
                      2 if "število=dvojina" in description else
                      3 if "število=množina" in description else -1,
            "owner_number": 1 if "število_svojine=ednina" in description else
                            2 if "število_svojine=dvojina" in description else
                            3 if "število_svojine=množina" in description else -1,
            "msd_tag_sl": msd_sl,
            "msd_tag_en": msd_en
        }
    return msd_dict


def read_msd_sl_specifications_to_list(filepath):
    category = list()
    attrs = list()
    vals = list()

    # https://raw.githubusercontent.com/mgrcar/OPA/6f0b45920f562db5a32aa23ac258ab4c759c2a51/Obeliks/LemmatizerTrain/MsdSpecsSloSloCodes.txt
    with open(filepath, "r") as f:
        position = 0
        for line in f.readlines():
            if "*" == line[0]:
                continue

            if line == "\n":
                line = line.strip()
                position += 1
            else:
                line = re.sub(r"\s+", " ", line)
                line = line.split(" ")

                if position == 1:
                    category.append(line[:3])
                    # print(line)
                elif position == 2:
                    attrs.append(line[:3])
                elif position == 3:
                    vals.append(line[:4])

    cats = dict()
    for cat in category:
        cats[cat[0]] = {"code": {
            "value": [tuple([cat[1], "_dummy_", cat[0]])],
            "position": 0,
        }}

    for att in attrs:
        cats[att[1]][att[0]] = {"position": int(att[2])}
        cats[att[1]][att[0]]["value"] = []

    # cats[val[3]][val[2]]["value"] = val[3]

    for val in vals:
        # print(val)
        cats[val[3]][val[2]]["value"].append((val[1], val[0], val[2]))

    return cats



# pridevnik
# {'position': 0, 'value': [('P', '_dummy_', 'pridevnik')]}
# {'position': 1,
#  'value': [('d', 'deležniški', 'vrsta'),
#            ('p', 'splošni', 'vrsta'),
#            ('s', 'svojilni', 'vrsta')]}
# {'position': 2,
#  'value': [('n', 'nedoločeno', 'stopnja'),
#            ('s', 'presežnik', 'stopnja'),
#            ('p', 'primernik', 'stopnja')]}
# {'position': 3,
#  'value': [('m', 'moški', 'spol'),
#            ('s', 'srednji', 'spol'),
#            ('z', 'ženski', 'spol')]}
# {'position': 4,
#  'value': [('d', 'dvojina', 'število'),
#            ('e', 'ednina', 'število'),
#            ('m', 'množina', 'število')]}
# {'position': 5,
#  'value': [('d', 'dajalnik', 'sklon'),
#            ('i', 'imenovalnik', 'sklon'),
#            ('m', 'mestnik', 'sklon'),
#            ('o', 'orodnik', 'sklon'),
#            ('r', 'rodilnik', 'sklon'),
#            ('t', 'tožilnik', 'sklon')]}
# {'position': 6, 'value': [('d', 'da', 'določnost'), ('n', 'ne', 'določnost')]}
# ta
# ('Zk-met', 'tega', 'zaimek vrsta=kazalni spol=moški število=ednina sklon=tožilnik', 'Pd-msa')



def get_dict(typ: str, refresh=False) -> dict:
    # pathlib.Path('tmp').mkdir(parents=True, exist_ok=True)

    if typ == "msd_sl":
        if Path(PTH_PKL_FILEPATH_MSD_SL).is_file() and not refresh:
            return load_dict_pickle(filepath=PTH_PKL_FILEPATH_MSD_SL)
        else:
            msd_dict = read_msd_xml_to_dict(PTH_DICT_MSD_SL_SPC, "sl")
            save_dict_pickle(msd_dict, PTH_PKL_FILEPATH_MSD_SL)
            return msd_dict
    elif typ == "msd_en":
        if Path(PTH_PKL_FILEPATH_MSD_EN).is_file() and not refresh:
            return load_dict_pickle(filepath=PTH_PKL_FILEPATH_MSD_EN)
        else:
            # sl and en are in the same file
            msd_dict = read_msd_xml_to_dict(PTH_DICT_MSD_SL_SPC, "en")
            save_dict_pickle(msd_dict, PTH_PKL_FILEPATH_MSD_EN)
            return msd_dict

    elif typ == "msd_tag_cats":
        if Path(PTH_PKL_FILEPATH_MSD_CATEGORIES).is_file() and not refresh:
            return load_dict_pickle(filepath=PTH_PKL_FILEPATH_MSD_CATEGORIES)
        else:
            cats = read_msd_sl_specifications_to_list(PTH_DICT_MSD_SPC_SLO_CODES)
            save_dict_pickle(cats, PTH_PKL_FILEPATH_MSD_CATEGORIES)
            return cats

    elif typ == "sloleks":
        if Path(PTH_PKL_FILEPATH_SLOLEKS_FORMS).is_file() \
                and Path(PTH_PKL_FILEPATH_SLOLEKS_PAIRS).is_file() \
                and not refresh:
            sloleks_forms = load_dict_pickle(
                filepath=PTH_PKL_FILEPATH_SLOLEKS_FORMS)
            sloleks_pairs = load_dict_pickle(
                filepath=PTH_PKL_FILEPATH_SLOLEKS_PAIRS)
        else:
            sloleks_forms, sloleks_pairs = read_sloleks_to_dict(PTH_DICT_SLOLEX_2_SL_TBL)
            save_dict_pickle(sloleks_forms, PTH_PKL_FILEPATH_SLOLEKS_FORMS)
            save_dict_pickle(sloleks_pairs, PTH_PKL_FILEPATH_SLOLEKS_PAIRS)
        return sloleks_forms, sloleks_pairs

    else:
        raise ValueError(
            "Dict tipe (@typ) to load must be one of 'msd_sl', 'sloleks'")



class DictHelper:
    def __init__(self, refresh=False):
        self.sloleks_forms, self.sloleks_pairs = get_dict(
            "sloleks", refresh=refresh)
        self.msd_dict_en = get_dict("msd_en", refresh=refresh)
        self.msd_dict_slo = get_dict("msd_sl", refresh=refresh)
        self.msd_tag_cats = get_dict("msd_tag_cats", refresh=refresh)

        # TODO: implement logger correctly
        # logger_level = logger.getEffectiveLevel()
        # logger_level = "debug" if logger_level == logging.DEBUG else "info" if logger_level == logging.INFO else ""
        # print(f"NOTE: DictHelper.py logging level is hardcoded to: {logger_level}")

    def word_exists(self, w):
        return w in self.sloleks_pairs

    def get_msd_info(self, msd_tag):
        msd_info = self.msd_dict_en[msd_tag]

        case = msd_info["case"]
        gender = msd_info["gender"]
        number = msd_info["number"]
        owner_number = msd_info["owner_number"]
        description = msd_info["description"]
        msd_tag_sl = msd_info["msd_tag_sl"]
        msd_tag_en = msd_info["msd_tag_en"]

        return case, gender, number, owner_number, description, msd_tag_sl, msd_tag_en

    def convert_msd_en_to_sl(self, msd_en):
        msd_info = self.msd_dict_en[msd_en]
        return msd_info["msd_tag_sl"]
    
    def convert_msd_sl_en(self, msd_sl):
        msd_info = self.msd_dict_slo[msd_sl]
        return msd_info["msd_tag_en"]

    def get_word_forms(self, word_searched) -> List[WordForm]:
        # returns:
        # [('Zk-zet',       # msd slo
        #   'to',           # word
        #   'zaimek vrsta=kazalni spol=ženski število=ednina sklon=tožilnik',   # description
        #   'Pd-fsa'), ...] # msd en
        logger.debug(f"{str(yellow(self.get_word_forms.__name__))} called with word: {word_searched}")

        try:
            if not self.word_exists(word_searched):
                return []
            basic_forms_set = self.sloleks_pairs[word_searched]
            # print(basic_forms_set)
            rez = list()
            for basic_form in basic_forms_set:
                # pprint(self.sloleks_forms[basic_form])
                for form in self.sloleks_forms[basic_form]:
                    # print(f"{form:8} {self.sloleks_forms[basic_form][form]:8s}")
                    word = self.sloleks_forms[basic_form][form]
                    # pprint(self.sloleks_forms[basic_form])

                    try:
                        descr = self.msd_dict_slo[form]
                        if descr:
                            rez.append(
                                WordForm(
                                    msd_slo=form,
                                    word_text=word,
                                    description=descr["description"],
                                    msd_en=descr["msd_tag_en"],
                                    examples=descr["examples"])
                                )
                    except KeyError as e:
                        logger.debug(f"Can not find {form} in self.msd_dict_slo")
                    # print("########################")
                    # pprint(self.msd_dict_slo[form])
            logger.debug(f"All word forms:\n{pformat(rez)}")
            return rez
        except Exception as e:
            logger.error(f"{str(red('Error on word:'))}, {(bold(word_searched))}, {e}")
            return []

    def change_tag(self, tag, position: int, letter):
        # change one letter in position(eg. Somei -> Somer)
        tag_new = tag
        tag_new_start, tag_new_end = tag_new[:position], tag_new[position+1:]
        tag_new = tag_new_start + letter + tag_new_end
        logger.debug(f"Changing tag from: {str(red(tag))} -> {str(green(tag_new))} (parts:, {tag_new_start} {letter} {tag_new_end})")
        return tag_new

    def change_tag_remove_animate(self, tag, position: int):
        tag_new = tag
        tag_new_start, tag_new_end = tag_new[:position], tag_new[position+1:]
        tag_new = tag_new_start + tag_new_end
        logger.debug(f"Removing potential animate from tag: {str(red(tag))} -> {str(green(tag_new))} (parts:, {tag_new_start} {tag_new_end})")
        return tag_new

    def change_msd_tag(self, tag, case=-1, number=-1, remove_animate=False):
        # eg. Somei -> Somer
        assert case in {-1, 1, 2, 3, 4, 5, 6}
        assert number in {-1, 1, 2, 3}
        
        logger.debug(f"{str(yellow(self.change_msd_tag.__name__))} called with args: tag: {tag}, case: {case}, num: {number}")
        tag_new = tag
        # print(len(self.msd_tag_cats))
        for c in self.msd_tag_cats:
            if tag[0] == self.msd_tag_cats[c]["code"]["value"][0][0]:

                msd_tag_cats_sorted = sorted(self.msd_tag_cats[c].values(), key=lambda x: x["position"])
                logger.debug(f"Options for tag: {str(green(tag))}\n{pformat(msd_tag_cats_sorted)}")
                for p in msd_tag_cats_sorted:

                    # remove animate (Sometn -> Somet)
                    if remove_animate and p["value"][0][2] == "živost":
                        tag_new = self.change_tag_remove_animate(tag_new, p["position"])

                    # set case
                    if case != -1 and p["value"][0][2] == "sklon":
                        tag_new_middle = "i" if case == 1 else \
                                         "r" if case == 2 else \
                                         "d" if case == 3 else \
                                         "t" if case == 4 else \
                                         "m" if case == 5 else \
                                         "o" if case == 6 else "!"
                        tag_new = self.change_tag(tag_new, p["position"], tag_new_middle)
                    else:
                        logger.debug("Skippink msd case changing because it is set to -1")

                    # set number
                    if number != -1 and p["value"][0][2] == "število":
                        tag_new_middle = "e" if number == 1 else \
                                         "d" if number == 2 else \
                                         "m" if number == 3 else "!"
                        tag_new = self.change_tag(tag_new, p["position"], tag_new_middle)
                    else:
                        logger.debug("Skippink msd number changing because it is set to -1")

        logger.debug(f"Changing tag from: {str(red(tag))} -> {str(green(tag_new))} (case: {case}, number: {number})")
        return tag_new

    def change_word_form_by_msd(self, word_current, tag_msd_current, case_wanted=-1, number_wanted=-1) -> WordForm:
        # for a word, find all its forms and then return only the wanted one (based on case and number)
        logger.debug(f"{str(yellow(self.change_word_form_by_msd.__name__))} called with word: "
                    f"{str(green(word_current))}, and tag: {str(bold(tag_msd_current))}")

        result: List[WordForm] = []
        word_all_forms = self.get_word_forms(word_current)
        # logger.debug(f"All word forms:\n{pformat(word_all_forms)}")
        if word_all_forms:
            tag_new = self.change_msd_tag(tag_msd_current, case=case_wanted, number=number_wanted)
            for form in word_all_forms:
                if form.msd_slo == tag_new:
                    # print(form)
                    result.append(form)

        if not result and word_all_forms:
            logger.debug("No correct word form found. Trying to remove 'animate' attribute from MSD tag and searching again.")
            tag_new = self.change_msd_tag(tag_msd_current, case=case_wanted, number=number_wanted, remove_animate=True)
            for form in word_all_forms:
                if form.msd_slo == tag_new:
                    # print(form)
                    result.append(form)

        if result:
            # TODO
            # if len(result) != 1:
            #     if not all([w.word_text == result[0].word_text for w in result]):
            #         logger.error(f"More than one form found for word {word_current}:\n{pformat(result)}")
            # assert len(result) == 1, f"More than one forms found for word {word_current}: {result}"
            word_new = result[0]
            do_not_replace_dict = {
                "s": "z",
                "z": "s",
                "kaj": "kva",
                "kva": "kaj",
                "bojo": "bodo",
                "bodo": "bojo",
            }
            if word_current in do_not_replace_dict and do_not_replace_dict[word_current] == word_new.word_text:
                return None
            
            logger.debug(f"Changing word to: {str(green(result[0].word_text))}")
            logger.debug(f"Returning new word form: {result[0]}")
            
            return word_new
        else:
            return None



def main():
    # msd_dict = get_dict("msd_sl")
    # sloleks_forms, sloleks_pairs = get_dict("sloleks")

    # pprint(msd_dict)
    # pprint(len(msd_dict))

    # sloleks_forms, sloleks_pairs = get_dict("sloleks", refresh=True)

    # for i in sloleks_pairs:
    #     if "vod" in i and "Sozmt" in i.lower():
    #         print(i)




    dh = DictHelper(refresh=False)

    msd_sl = "Ppnzmr"
    msd_en = dh.convert_msd_sl_en(msd_sl)
    test = dh.get_msd_info(msd_en)
    test = test[0] if test else None
    
    
    
    print(test)
    exit()

    word_orig_text = "bodo"
    word_orig_msd_sl = "Gp-ptm-n"
    case_wanted = -1
    number_wanted = 3
    
    # word_all_forms = dh.get_word_forms(word_orig_text)
    # pprint(word_all_forms)
    
    rez = dh.change_word_form_by_msd(word_orig_text, word_orig_msd_sl, case_wanted=case_wanted, number_wanted=number_wanted)
    print(rez)
    exit()

    print("###########")
    word_new = dh.change_word_form_by_msd(word_orig_text, word_orig_msd_sl, case_wanted=case_wanted, number_wanted=number_wanted)
    pprint(word_new)
    
    print("$$$$$$$$$$")
    word_new = dh.change_word_form_by_msd(word_orig_text, word_orig_msd_sl, case_wanted=3, number_wanted=number_wanted)
    pprint(word_new)
    
    
    print("$$$$$$$$$$")
    word_new = dh.change_word_form_by_msd("osnovam", "Sozmd", case_wanted=1, number_wanted=-1)
    pprint(word_new)
    exit()
    
    word_orig_text = "osnovah"
    word_orig_msd_sl = "Sozmm"
    case_wanted = 3
    number_wanted = -1
    
    # word_all_forms = dh.get_word_forms(word_orig_text)
    # pprint(word_all_forms)
    
    word_new = dh.change_word_form_by_msd(word_orig_text, word_orig_msd_sl, case_wanted=case_wanted, number_wanted=number_wanted)
    pprint(word_new)
    
    
    


    # word_new = dh.change_word_form_by_msd("tega", "Zk-met", case_wanted=3, number_wanted=3)
    # word_new = dh.change_word_form_by_msd("čaj", "Sometn", case_wanted=2, number_wanted=1)
    # print(word_new)

    tag = "Zk-met"
    tag_new = dh.change_msd_tag(tag, case=3, number=3)
    print(tag_new)
    
    print(dh.convert_msd_en_to_sl("Pd-mpd"))

    # for w in ["tem", "ta", "tega", "temu"]:
    #     word_all_forms = dh.get_word_forms(w) #, form_searched="Zk-sei", filter_forms=True)

    #     print(green(w))
    #     # pprint(word_all_forms)

    #     for form in word_all_forms:
    #         if form[0] == tag_new:
    #             # print(green("########"))
    #             print(form)



    # print(dh.get_word_from_lemma_and_msd("ta", ""))

    # print()
    # basic_form = sloleks_pairs["Žana"]
    # for form in sloleks_forms[basic_form]:
    #     desc = msd_dict[form]["description"]
    #     print(f"{form} {sloleks_forms[basic_form][form]:8s} {desc}")

    # def printtt(w):
    #     w_basic_form = sloleks_pairs[w]
    #     print(f"{bold(green(w))} - {w_basic_form}:")
    #     for i in sloleks_forms[w_basic_form]:
    #         desc = msd_dict[i]["description"]
    #         w_pair = sloleks_forms[w_basic_form][i]
    #         print(f"    {w_pair:<10} : {i:>8} - {desc}")

    # # for w in list(sloleks_forms):
    # #     try:
    # #         w_basic_form = sloleks_pairs[w]
    # #         if w != w_basic_form:
    # #             printtt(w)
    # #     except Exception:
    # #         pass

    # printtt("Žan-Slmei")
    # # printtt("Žana")

    # # msd_dict = read_msd_xml_to_dict()
    # # save_dict_pickle(msd_dict, "saved_dict_msd_sl.pkl")

    # # msd_dict = load_dict_pickle("saved_dict_msd_sl.pkl")
    # assert "Somei" in msd_dict

    # cats = read_msd_sl_specifications_to_list()
    # match = "Slzmr"
    # change_msd_tag(cats, match)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)
    
    main()
