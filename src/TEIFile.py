import re
from bs4 import BeautifulSoup
from dataclasses import dataclass
import logging
import glob
from tqdm import tqdm
import multiprocessing as mp
from curtsies.fmtfuncs import green, bold, yellow, red



def elem_to_text(elem, default=''):
    if elem:
        return elem.getText()
    else:
        return default


def read_tei(tei_file):
    with open(tei_file, 'r') as tei:
        soup = BeautifulSoup(tei, 'xml')
        return soup
    raise RuntimeError('Cannot generate a soup from the input')

categories = {
            "tisk": "SSJ.T",
            "knjižno": "SSJ.T.K",
            "leposlovno": "SSJ.T.K.L",
            "strokovno": "SSJ.T.K.S",
            "periodično": "SSJ.T.P",
            "časopis": "SSJ.T.P.C",
            "revija": "SSJ.T.P.R",
            "drugo": "SSJ.T.D",
            "internet": "SSJ.I"
        }


class Word:
    def __init__(self, w, is_space=False):
        self.is_space = is_space
        if self.is_space:
            self.word = " "
            self.lemma = " "
            self.msd = " "
        else:    
            self.word = w.text.strip()
            self.lemma = w.get("lemma")
            self.msd = w.get("msd")

    def __str__(self):
        return self.word
    
    def __repr__(self):
        return f"[{self.msd}]{self.word}"
    

def set_category_filter(cats: list):
    intersection = set([categories[x] for x in cats]).intersection(set([categories[x] for x in categories]))
    if len(intersection) == 0:
        raise ValueError("category filter intersection is empty! categories dict may not be completely set up. Categories: ", 5)
    return intersection


def get_sentence_from_list(sentence, filter_form=""):
    sentence_with_words = []
    # sentence_with_spaces = []
    # sentence_to_print = ""
    
    for w in sentence:
        # if repr(w) == "<S/>":
        #     word = Word("", is_space=True)
        # else:
        word = Word(w, is_space=repr(w) == "<S/>")
        sentence_with_words.append(word)
        
        # if word.is_space:
        #     sentence_with_spaces.append(" ")
            # sentence_to_print += " "
        

        # to_append = str(word)
        # sentence_with_spaces.append(to_append)
        
        # sentence_to_print += bold(green(to_append)) if word.msd == "Somei" else to_append
            
    # print(sentence_to_print)
    # sentence_with_spaces = [f"{str(w)}[{w.msd}]" for w in sentence_with_words]
    # sentence = "".join(sentence_with_spaces)

    # for w in sentence_with_words:
    #     if "Somei" == w.msd:
    #         print("###", green(str(w)), sentence)
    #         break

    REPLACEMENTS_ENABELED = False
    if REPLACEMENTS_ENABELED:
        replacements = [
            # (r'(?:[\.,]\s*)+', '.'),
            (r'\d{2}\.\s*\d{2}\.\s*\d{4}', ''),

            (r'–', '-'),
            (r'[»«\'“”‘’]', '"'),
            
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
            sentence = re.sub(old, new, sentence)

    # print(sentence)
    # if " ." in sentence:
    #     print(red(sentence))
    # if " ," in sentence:
    #     print(green(sentence))
    # if ".." in sentence: 
    #     print(yellow(sentence))
    
    if len(filter_form) == 0 \
        or filter_form in [w.msd for w in sentence_with_words]:
        # print(sentence_to_print)
        # print(sentence)
        # return sentence_with_words
        return sentence

    return []

def process_sentence(sentence, filter_form=""):
    sentence = sentence.find_all(["w", "c", "S"])
    if sentence.count('\n') / len(sentence) > 0.05 \
            or sum(c.text.isdigit() for c in sentence) / len(sentence) > 0.03 \
            or sentence.count(':') / len(sentence) > 0.01 \
            or sentence.count('(') / len(sentence) > 0.01:
        return
            
    if len(sentence) < 10:
        return
    kek = get_sentence_from_list(sentence, filter_form=filter_form)
    return kek


# https://komax.github.io/blog/text/python/xml/parsing_tei_xml_python/
class TEIFile(object):
    categories_filter: set # = {'SSJ.T.K.L', 'SSJ.T', 'SSJ.T.P.R', 'SSJ.T.D', 'SSJ.I', 'SSJ.T.K', 'SSJ.T.K.S', 'SSJ.T.P.C', 'SSJ.T.P'}

    def __init__(self, filename):
        self.filename = filename
        self.soup = read_tei(filename)
        self._text = None
        self._title = ''
        self._abstract = ''

    def filter_category(self):
        cats = self.text_type
        # print(set(cats), self.categories_filter)
        if set(cats) & self.categories_filter:
            return True
        return False
    
    @property
    def doi(self):
        idno_elem = self.soup.find('idno', type='DOI')
        if not idno_elem:
            return ''
        else:
            return idno_elem.getText()
        
    @property
    def num_words(self) -> int:
        # print(self.soup)
        num_words = self.soup.find("teiHeader").find("fileDesc").find("extent")
        num_words = num_words.text.strip("besed").strip()
        num_words = int(num_words)
        return num_words

    @property
    def source(self):
        src = self.soup.find("teiHeader").find("fileDesc").find("sourceDesc").find("bibl")
        title = elem_to_text(src.find("title"))
        date = elem_to_text(src.find("date"))

        
    @property
    def text_type(self):
        """get text category hashtags

        Returns:
            List[str]: list of category hashtags
        """
        try:
            cat_ref = self.soup.find("profileDesc")
            cat_ref = cat_ref.find("textClass")
            cat_ref = cat_ref.find("catRef")
            return [x.strip().strip("#") for x in cat_ref["target"].split(" ")]
        except Exception:
            logging.exception("Content type not found")
            return []

    @property
    def title(self):
        if not self._title:
            self._title = self.soup.title.getText()
        return self._title

    @property
    def abstract(self):
        if not self._abstract:
            abstract = self.soup.abstract.getText(separator=' ', strip=True)
            self._abstract = abstract
        return self._abstract

    # @property
    # def authors(self):
    #     authors_in_header = self.soup.analytic.find_all('author')

    #     result = []
    #     for author in authors_in_header:
    #         persname = author.persname
    #         if not persname:
    #             continue
    #         firstname = elem_to_text(persname.find("forename", type="first"))
    #         middlename = elem_to_text(persname.find("forename", type="middle"))
    #         surname = elem_to_text(persname.surname)
    #         # person = Person(firstname, middlename, surname)
    #         # result.append(person)
    #     return result


    def text(self, filter_form=""):
        if not self._text:
            sentences = []
            body = self.soup.find("text").find("body")
            for paragraph in body.find_all("p"):
                for sentence in paragraph.find_all("s"):
                    s = process_sentence(sentence, filter_form=filter_form)
                    if s:
                        sentences.append(s)
            # div is neither an appendix nor references, just plain text.
            # if not body.get("type"):
            #     div_text = body.get_text(separator=' ', strip=True)
            #     divs_text.append(div_text)

            # plain_text = " ".join(divs_text)
            self._text = sentences
        return self._text


def process(tei_doc):
    tei = TEIFile(tei_doc)
    if tei.filter_category():
        # print(tei_doc)
        # print(f"Num words: {tei.num_words: 5d}, type: {tei.text_type}")
        # print(tei.text)
        sentence = tei.text("Somei")
        print(sentence)
        # if "Somei" in [w.msd for w in sentence]:
        #     return sentence
    return []
    # print(f"Num words: {tei.text_type}")
    # todo: write to file https://stackoverflow.com/questions/13446445/python-multiprocessing-safely-writing-to-a-file


def main():
    # teis = glob.glob("/home/timotejp/dipl/ccGigafida1.0/ccGigafidaV1_0/F0024824.xml")
    gf_tei_xmls = glob.glob("corpus/ccGigafidaV1_0/*.xml")
    
    TEIFile.categories_filter = set_category_filter(cats=["strokovno"])

    number_of_cores = 4  # int(mp.cpu_count()*3/4)
    logging.info(f"Using {number_of_cores} of cores.")
    with mp.Pool(processes=number_of_cores) as pool:
        for _ in tqdm(pool.imap(process, gf_tei_xmls),
                      total=len(gf_tei_xmls),
                      desc="Processing tei xmls"):
            pass    # a hacky way to go through all the files and use tqdm
    
    
if __name__ == "__main__":
    main()




###############################################################################
# from tei_reader import TeiReader
# import os

# reader = TeiReader()

# fileNames = glob.glob("ccGigafida1.0/ccGigafidaV1_0/F0000003.xml")
# for fileName in fileNames:

#     corpora = reader.read_file(fileName)  # or read_string
#     # # print(corpora.text)
#     for i in corpora.documents:
#         print(i.tostring(lambda x, text: str(
#             list(a.key + '=' + a.text for a in x.attributes)) + text))
#         break
#     # show element attributes before the actual element text
#     # print(corpora.tostring(lambda x, text: str(list(a.key + '=' + a.text for a in x.attributes)) + text))

#     #if file_in_counter > 15:
#     #  break
#     #file_in_counter += 1
