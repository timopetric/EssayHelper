import os
from glob import glob
import pathlib
import pickle
from typing import List
from constants import *
from collections import namedtuple
import pandas as pd


Thresholds = namedtuple("Thresholds", [
    "direct_replacement_if_same_lemma",
    "skip_if_first_sugg_below",
    "case12", "case13", "case14", "case15", "case16",
    "case21", "case23", "case24", "case25", "case26",
    "case31", "case32", "case34", "case35", "case36",
    "case41", "case42", "case43", "case45", "case46",
    "case51", "case52", "case53", "case54", "case56",
    "case61", "case62", "case63", "case64", "case65",
    "number12", "number13",
    "number21", "number23",
    "number31", "number32"
    ])
EvalParams = namedtuple("EvalParams", ["case_from_to", "number_from_to", "num_wrong_words"])
WordFormChangeCase = namedtuple("WordFormChangeCase", ["case_from", "case_to"])
WordFormChangeNumber = namedtuple("WordFormChangeNumber", ["number_from", "number_to"])
WordForm = namedtuple("WordForm", ["msd_slo", "word_text", "description", "msd_en", "examples"])
WrongWord = namedtuple("WrongWord", ["inx_words", "word_text"])
Location = namedtuple("Location", ["url", "path", "extract", "md5"])
ConfusionMatrix = namedtuple("ConfusionMatrix", ["tp", "fp", "fn", "tn"])


def get_glob_file_iteratior(glob_path):
    def get_filepath_name_split(fullpath):
        """returns (filepath, filename)""" 
        return (os.path.dirname(fullpath), os.path.basename(fullpath))
    return map(get_filepath_name_split, glob(glob_path))


def load_test_input_texts() -> List[str]:
    input_texts = list()
    for filepath in sorted(glob(PTH_TEST_INPUT_TEXTS_FOL+"/input_*.txt")):
        with open(filepath, "r") as f:
            input_texts.append(f.read())
    if not input_texts:
        raise ValueError(f"There are no files like 'input_*.txt' in folder {PTH_TEST_INPUT_TEXTS_FOL}")
    return input_texts

# used in slobertaMCD allegedly
def pkl_save_to(data, path):
    assert ".pkl" in path[-4:], "file must end with '.pkl'"
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def pkl_read_from(path):
    assert ".pkl" in path[-4:], "file must end with '.pkl'"
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# used in DictHelper
def save_dict_pickle(w_dict: dict, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(w_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_pickle(filepath) -> dict:
    with open(filepath, 'rb') as handle:
        w_dict = pickle.load(handle)
        return w_dict

# used in SloHelper
def pickle_dump_save(to_save, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load_dict(filepath):
    with open(filepath, 'rb') as handle:
        loaded = pickle.load(handle)
        return loaded

def get_eval_params_str(ep: EvalParams):
    # # for case and numbers
    cases = "-".join([f"c{c_from}{c_to}" for c_from, c_to in ep.case_from_to])
    numbers = "-".join([f"n{n_from}{n_to}" for n_from, n_to in ep.number_from_to])
    wrong_words = f"nww{ep.num_wrong_words}"
    eval_params_str = f"{cases}_{numbers}_{wrong_words}"
    return eval_params_str

def find_newtest_result():
    # newtest file should have the last saved result.csv - with all 4000 iterations of sherpa
    
    path_main = PTH_SHERPA_THRERSHOLDS
    paths = ['/c*nww*',
             '/_n*nww*',
             '/direct_replacement_if_same_lemma___skip_if_first_sugg_below']
    for path in paths:
        list_of_folders = glob(path_main + path)
        for folder in list_of_folders:
            parrent_folder = pathlib.PurePath(folder).name
            list_of_files = glob(os.path.join(folder, "*/results.csv"))
            latest_file = max(list_of_files, key=os.path.getctime)
            yield latest_file, parrent_folder

def get_best_sherpa_result(top_k=5, num_iterations=4000, verbose=True):
    if not verbose:
        print("Thresholds(")
    
    for filepath_results, parrent_folder in find_newtest_result():
        df = pd.read_csv(filepath_results)
        if verbose and not len(df) == num_iterations*2:
            print(f"\t!!!!!!!!!! It looks like results: {parrent_folder} are not yet finished. It has {len(df)} lines, should have {num_iterations}.")
        df.sort_values("Objective", inplace=True, ascending=False)
        
        num_correct_replacements = int(df.head(1)["num_correct_replacements"])
        num_sentences = int(df.head(1)["num_sentences"])
        num_wrong_words = int(df.head(1)["num_wrong_words"])
    
        if "direct_replacement_if_same_lemma" in df:
            mean_direct_replacement_if_same_lemma = df.head(top_k)["direct_replacement_if_same_lemma"].mean()
            mean_skip_if_first_sugg_below = df.head(top_k)["skip_if_first_sugg_below"].mean()
            print(f"\tdirect_replacement_if_same_lemma={mean_direct_replacement_if_same_lemma:.4f},")
            print(f"\tskip_if_first_sugg_below={mean_skip_if_first_sugg_below:.4f},")
    
        if "case" and "number" in df:    
            mean_case = df.head(top_k)["case"].mean()
            mean_number = df.head(top_k)["number"].mean()
            # print(df)
            
            if parrent_folder.startswith("c"):
                c_from = int(parrent_folder[1])
                c_to = int(parrent_folder[3])
                if not verbose:
                    # because we set fake cases to c_to case when training 
                    # we must invert the selection
                    print(f"\tcase{c_to}{c_from}={mean_case:.3f},")
                    print(f"\tcase{c_to}{c_from}_num={mean_number:.3f},")
            # elif parrent_folder.startswith("_n"):
            #     n_from = int(parrent_folder[2])
            #     n_to = int(parrent_folder[4])
            #     if not verbose:
            #         print(f"\tnumber{n_from}{n_to}={mean_number:.3f},")
            # else:
            #     print("ERROR, no startswith c or _n found.")

        if verbose:
            print(f"{parrent_folder}\n\tlen: {len(df)}\n\tpath: {filepath_results}\n\tc: {mean_case*100:.2f}%, n: {mean_number*100:.2f}%")
            print(f"\tnum_sentences = num_wrong_words: {num_sentences}")
            # print(f"\tnum_wrong_words: {num_wrong_words}")
            print(f"\tnum_correct_replacements: {num_correct_replacements}")
    if not verbose:
        print(")")



if __name__ == "__main__":
    pass
    # a = load_test_input_texts() 
    # print(len(a))
    # print(a[3])
    
    # get_best_sherpa_result(verbose=False)
    
    # ep = EvalParams(case_from_to=[(4,2)], number_from_to=[(1,3)], num_wrong_words=1)
    # print(getattr(ep, "case_from_to"))
    # print(get_eval_params_str(ep))



    # print(getattr(thresholds, "case12"))
    