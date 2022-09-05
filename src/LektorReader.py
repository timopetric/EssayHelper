from pprint import pprint
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass
import logging
import glob
from tqdm import tqdm
import multiprocessing as mp
from curtsies.fmtfuncs import green, bold, yellow, red
import random
from constants import PTH_SOLAR_CORPUS
from bs4 import BeautifulSoup

from DictHelper import DictHelper
from constants import PTH_LEKTOR_OUT, PTH_LEKTOR_OUT_TEST, PTH_LEKTOR_OUT_TRAIN


dh = DictHelper()

def lektor_preprocess(lektor_path_orig, lektor_path_preprocessed):
    # skip_next_space = False

    with open(lektor_path_orig, "r") as f1, \
         open(lektor_path_preprocessed, "w") as f2:

        lines = f1.readlines()
        for inx, line_curr in enumerate(lines):
            if inx == len(lines)-1:
                # skip first
                f2.write(line_curr)

            else:
                line_curr = line_curr.strip()
                line_next = lines[inx+1].strip()

                if "<s1/>" == line_curr:
                    # start of the sentence
                    line_curr = "<s0>"
                    f2.write(f"{line_curr}\n")
                    continue

                elif "<s0/>" == line_curr:
                    # end of the sentence
                    line_curr = "</s0>"
                    f2.write(f"{line_curr}\n")
                    continue                

                # # print(line_prev, line_curr)
                # if skip_next_space:
                #     # skip this space (as insructed in prev iteration)
                #     assert line_curr == "<S/>", f"cant skip non existent space for line: {line_curr}"
                #     # print("skipping")
                #     skip_next_space = False
                #     continue

                # elif line_curr == "<S/>" and "<lekt" == line_next[:5]:
                #     # skip this space
                #     continue

                # elif "</lekt" == line_curr[:6] and line_next == "<S/>":
                #     # skip next space
                #     # print("## skipping")
                #     skip_next_space = True
                #     f2.write(f"{line_curr}\n")
                #     continue

                f2.write(f"{line_curr}\n")
            
    print("Done processing lektor corpus.")




def compile_lekt_words(w, pgh_str, w_forms_dict: dict, analize_corrections=False):
    # print("### compile_lekt_words called, sent_str:", sent_str, "\n", "word: ", yellow(str(w)))
    # str(f"ins{lekt_stage}")
    lekt_elements = w.find_all(["w", "s", "c", "lekt1", "lekt2", "lekt3", "lekt4"], recursive=False)
    for w in lekt_elements:
        pgh_str, w_forms_dict = process_w_soup(w, pgh_str, w_forms_dict, analize_corrections=analize_corrections)
    return pgh_str, w_forms_dict


def process_w_soup(w, pgh_str, w_forms_dict: dict, analize_corrections=False):
    # pgh_str = ""
    xml_tag_start_part = str(w).strip()[:10]
    if "<lekt" in xml_tag_start_part[:5]:
        # print("$$$ <lekt found, calling compile_lekt_words, w start", green(xml_tag_start_part))
        tags = ["w", "s", "c", "ins1", "ins2", "ins3", "ins4"]
        if analize_corrections:
            tags += ["del1", "del2", "del3", "del4"]
        for w2 in w.find_all(tags, recursive=False):
            pgh_str, w_forms_dict = compile_lekt_words(w2, pgh_str, w_forms_dict, analize_corrections=analize_corrections)
    
    elif "<w" in xml_tag_start_part[:2]:
        # print("word", w)

        if len(pgh_str) > 0 and pgh_str[-1].isalpha():
            # fix potentially missing spaces
            pgh_str = pgh_str + " "

        pgh_str += w.text.strip()
        
        if w.has_attr("msd"):
            msd_tag = str(w["msd"])
            if msd_tag in dh.msd_dict_slo:
                msd_info = dh.msd_dict_slo[msd_tag]
                if msd_info:
                    case = msd_info["case"]
                    number = msd_info["number"]
                    if case != -1:
                        w_forms_dict[str(f"c{case}")] += 1
                    if number != -1:
                        w_forms_dict[str(f"n{number}")] += 1
                    # print(f"c{case}, n{number}")

    elif "<s></s>" in xml_tag_start_part[:7]:
        # print("space", w)
        if len(pgh_str) > 0 and pgh_str[-1] in ["("]:
            return pgh_str, w_forms_dict
        pgh_str += " "
        
    elif "<c" in xml_tag_start_part[:2]:
        # print("comma", w)
        if len(pgh_str) > 0 and pgh_str[-1] == " " and w.text.strip() in [".", ",", ")", ";"]:
            pgh_str = pgh_str[:-1]
        pgh_str += w.text.strip()

    return pgh_str, w_forms_dict


def format_line_w_forms_dict(w_forms_dict):
    string = f""
    string += f"c1:{w_forms_dict['c1']}|"
    string += f"c2:{w_forms_dict['c2']}|"
    string += f"c3:{w_forms_dict['c3']}|"
    string += f"c4:{w_forms_dict['c4']}|"
    string += f"c5:{w_forms_dict['c5']}|"
    string += f"c6:{w_forms_dict['c6']}|"
    string += f"n1:{w_forms_dict['n1']}|"
    string += f"n2:{w_forms_dict['n2']}|"
    string += f"n3:{w_forms_dict['n3']}"
    return str(string)

def extract_line_to_w_forms_dict(line) -> dict:
    w_forms_dict = {"c1": 0, "c2": 0, "c3": 0, "c4": 0, "c5": 0, "c6": 0, "n1": 0, "n2": 0, "n3": 0,}
    line_split = line.split("|")
    # TODO: Change to for loop
    w_forms_dict["c1"] = int(line_split[0].split(":")[1])
    w_forms_dict["c2"] = int(line_split[1].split(":")[1])
    w_forms_dict["c3"] = int(line_split[2].split(":")[1])
    w_forms_dict["c4"] = int(line_split[3].split(":")[1])
    w_forms_dict["c5"] = int(line_split[4].split(":")[1])
    w_forms_dict["c6"] = int(line_split[5].split(":")[1])
    w_forms_dict["n1"] = int(line_split[6].split(":")[1])
    w_forms_dict["n2"] = int(line_split[7].split(":")[1])
    w_forms_dict["n3"] = int(line_split[8].split(":")[1])
    return w_forms_dict

def extract_sentence_and_w_forms_dict_from_line(line):
    line = line.split("#-#-#-#")
    w_forms_dict = extract_line_to_w_forms_dict(line[0])
    sent = line[1]
    return sent, w_forms_dict

def lektor_extract_sentences(lektor_path_preprocessed, lektor_path_out, analize_corrections=None):
    PARAGRAPH_MIN_LEN = 32

    with open(lektor_path_preprocessed, "r") as f_in, open(lektor_path_out, "w") as f_out:
        corpus_str = f_in.read()
        for doc_inx, doc in tqdm(enumerate(corpus_str.split("</text>")[:-1]), total=30):
            doc += "</text>"

            # if doc_inx == 1: break

            doc_soup = BeautifulSoup(doc, 'lxml')
            head = doc_soup.find("head")
            text = doc_soup.find("text")

            paragraphs = []
            for paragraph in text.find_all("p"):
                # pgh_str = ""
                for sentence in paragraph.find_all("s0"):
                    snt_str = ""
                    w_forms_dict = {"c1": 0, "c2": 0, "c3": 0, "c4": 0, "c5": 0, "c6": 0, "n1": 0, "n2": 0, "n3": 0,}

                    sent_elements = sentence.find_all(["w", "s", "c", "lekt1"], recursive=False)
                    for w in sent_elements:
                        snt_str, w_forms_dict = process_w_soup(w, snt_str, w_forms_dict, analize_corrections=analize_corrections)

                    # pprint(w_forms_dict)
                    # exit()

                    if len(snt_str) > PARAGRAPH_MIN_LEN and \
                        not ((sum(c.isalpha() for c in snt_str) / len(snt_str) < 0.7) or
                             (sum(c.isdigit() for c in snt_str) / len(snt_str) > 0.03) or
                             (snt_str.count(':') / len(snt_str) > 0.01) or
                             (snt_str.count('(') / len(snt_str) > 0.01)
                            ) and \
                        snt_str.strip()[0].isupper() and snt_str.strip()[-1] == ".":

                        REPLACEMENTS_ENABELED = True
                        if REPLACEMENTS_ENABELED:
                            replacements = [
                                # (r'(?:[\.,]\s*)+', '.'),
                                (r'\d{2}\.\s*\d{2}\.\s*\d{4}', ''),

                                (r'–', '-'),
                                (r'[»«\'“”‘’]', '"'),
                                (r"''", '"'),
                                
                                (r'[#$€@•&ø×¹…\[\]]', ''),
                                # (r'[#$€:;@\'"“”•«»%&ø×¹…’‘°\[\]]', ''),
                                
                                (r'"/?\d+', '"'),
                                (r'\./?\d+', '.'),
                                (r'[áä]', 'a'),
                                (r'[ÁÄ]', 'A'),
                                
                                (r'[éë]', 'e'),
                                (r'[ÉË]', 'E'),
                                
                                (r'í', 'i'),
                                (r'Í', 'I'),
                                
                                (r'[öó]', 'o'),
                                (r'[ÖÓ]', 'O'),
                                
                                (r'úü', 'u'),
                                (r'ÚÜ', 'U'),

                                (r'ć', 'c'),
                                (r'Ć', 'C'),
                                
                                # (r'\s+([/])\s+', r'\1'),                 # remove spaces before and after "/"
                                # (r'((?:\s|$)[(])\s',  r'\1'),            # remove spaces after "("
                                # (r'\s+([\.\,\!\?\\)](?:\s|$))', r'\1'),  # remove spaces before punctuations
                                (r'\s+', ' '),                             # replace multiple spaces w/ one
                            ]
                            for old, new in replacements:
                                snt_str = re.sub(old, new, snt_str)
                            
                        line_to_write = str(f"{format_line_w_forms_dict(w_forms_dict)}#-#-#-#{snt_str.strip()}")
                        paragraphs.append(line_to_write)
                        if not analize_corrections:
                            f_out.write(f"{line_to_write}\n")

            pprint(paragraphs[:3], width=120)


def analize(lektor_path_out):
    with open(lektor_path_out, "r") as f:
        lines = f.readlines()
        num_lines = len(lines)
        ends_with_punctuation = 0
        starts_with_capital = 0
        both = 0
        for l in lines:
            if l.strip()[0].isupper():
                starts_with_capital += 1
            if l.strip()[-1] == ".":
                ends_with_punctuation += 1
            if l.strip()[0].isupper() and l.strip()[-1] == ".":
                both += 1

        print(f"All: {num_lines}")
        print(f"Cap start: {starts_with_capital}")
        print(f"Of all: {starts_with_capital/num_lines*100:7.3f}")
        
        print(f"\n. at end: {ends_with_punctuation}")
        print(f"Of all: {ends_with_punctuation/num_lines*100:7.3f}")

        print(f"\nBoth: {both}")
        print(f"Of all: {both/num_lines*100:7.3f}")

def write_lines_list_to_file(filepath, lines):
    assert isinstance(lines, list)
    
    with open(filepath, "w") as f:
        for l in lines:
            f.write(l)
            
def split_train_test(lektor_path_full=PTH_LEKTOR_OUT, lektor_path_train=PTH_LEKTOR_OUT_TRAIN, lektor_path_test=PTH_LEKTOR_OUT_TEST):
    # analize(lektor_path_out)
    with open(lektor_path_full, "r") as lektor_in:
        lines = list(lektor_in.readlines())

    lines = sorted(lines, key=lambda l: len(l))

    lines_train = []
    lines_test = []
    for i, line in enumerate(lines):
        if i % 5 == 0:
            lines_test.append(line)
        else:
            lines_train.append(line)
            
    random.shuffle(lines_train)
    random.shuffle(lines_test)
    
    write_lines_list_to_file(lektor_path_train, lines_train)
    write_lines_list_to_file(lektor_path_test, lines_test)

def solar_analisys():
    path = PTH_SOLAR_CORPUS
    with open(path, "r") as f:
        solar = f.read()
        
        
    dh = DictHelper(refresh=False)

    def get_msd_info(msd_sl):
        # msd_sl = "Ppnzmr"
        msd_en = dh.convert_msd_sl_en(msd_sl)
        msd_info = dh.get_msd_info(msd_en)
        return msd_info[0] if msd_info else None
    

    bs = BeautifulSoup(solar, features="lxml")
    essays = bs.find_all(["div"]) #, {"xml:id" : "sklon-rt"})
    num_essays = len(essays)
    num_essays_rt = 0
    num_essays_dm = 0
    num_essays_dm_rt = 0
    
    from tqdm import tqdm
    
    napake = dict()
    napak_num = 0
    
    for essay in tqdm(essays):
    
        # corrections_kat = essay.find([f"u{i}" for i in range(1, 6)], {"podtip": "KAT"})
        corrections_kat = essay.find_all(["u1"], {"podtip": "KAT"})
        for c in corrections_kat:
            kat = c["kat"]
            if kat not in napake:
                napake[kat] = 0
            napake[kat] += 1  # {"kat" : "sklon-rt"}
            napak_num += 1
    
        corrections_rt = essay.find(["u1"], {"kat" : "sklon-rt"})
        if corrections_rt:
            num_essays_rt += 1
            # print(corrections_rt)

        corrections_dm = essay.find(["u1"], {"kat" : "sklon-dm"})
        if corrections_dm:
            num_essays_dm += 1
            # print(corrections_dm)
        
        if corrections_dm or corrections_rt:
            num_essays_dm_rt += 1
        # for u in corrections_rt:
        #     words_first = u.find_all(["w1"])
        #     for w in words_first:
        #         # print(w["ana"])
        #         msd_tag, text = w["ana"][4:], w.text
        #         case = get_msd_info(msd_tag)
        #         print(msd_tag, text, case)
        #         if case == 4:
        #             pass
    
    print(num_essays)
    print(num_essays_rt, f"{num_essays_rt/num_essays*100:.3f}")
    print(num_essays_dm, f"{num_essays_dm/num_essays*100:.3f}")
    print(num_essays_dm_rt, f"{num_essays_dm_rt/num_essays*100:.3f}")
    
    print("###################")
    print(napake)
    for n in napake:
        print(n, napake[n], f"{napake[n]/napak_num*100:.3f}%")
    
    # print(corrections)



def main():
    lektor_path_orig = "dictionaries/lektor_corpus/LEKTOR.186"
    lektor_path_preprocessed = "tmp/LEKTOR_PREPROCESSED.186"

    # lektor_preprocess(lektor_path_orig, lektor_path_preprocessed)
    # lektor_extract_sentences(lektor_path_preprocessed, PTH_LEKTOR_OUT, analize_corrections=True)
    
    # split_train_test(lektor_path_full=PTH_LEKTOR_OUT,
    #                  lektor_path_train=PTH_LEKTOR_OUT_TRAIN,
    #                  lektor_path_test=PTH_LEKTOR_OUT_TEST)

    solar_analisys()

    # echo "$(tail -10000 lektor_out.txt)" > lektor_out_test.txt 
    # echo "$(head -18744 lektor_out.txt)" > lektor_out_train.txt 
    # wc lektor_out*

if __name__ == "__main__":
    main()
