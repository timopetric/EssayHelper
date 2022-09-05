from DictHelper import get_dict
from curtsies.fmtfuncs import green, bold, yellow, red
from codecs import getdecoder
import classla
from classla.utils.conll import CoNLL
import toma
from classla.models.common.doc import Document as DocumentClassla
from typing import List
from tqdm import tqdm
from datetime import datetime


tokenizer = classla.Pipeline('sl', processors="tokenize", use_gpu=True)
nlp = classla.Pipeline('sl', processors="tokenize,pos,lemma",
                        tokenize_pretokenized=True,
                        use_gpu=True,
                        tokenize_batch_size=5000,
                        pos_batch_size=10000,
                        lemma_batch_size=10000)


def run_batch(batch_size: int, nlp: classla.Pipeline, data: List[List[str]]
              ) -> List[DocumentClassla]:
    # So that we can see what the batch size changes to.
    print(batch_size)
    rez = []
    for i in tqdm(range(0, len(data)-batch_size, batch_size)):
        processed_classla = nlp(data[i:i+batch_size])
        rez += [sent.words for sent in processed_classla.sentences]
    return rez


sents = [
    "Ni uspel popiti kave.",
    "Ni uspel popiti pijače.",
    "Ni uspel popiti vode.",
    "Ni uspel popiti čaj.",
    "Ni uspel popiti alkohola.",
    "Lahko bi, ampak to ni storil.",
    "Lahko bi, ampak tega ni storil.",
]
sents_tokenized = []

# classla.download('sl') # download standard
sents_tokenized = [
    [x.get("text") for x in tokenizer(i).to_dict() [0][0]]
    for i in sents]
sents_tokenized *= 10000000

print("Tokenization DONE, num of sents_tokenized:", len(sents_tokenized))





print("Starting pipeline processing")
start = datetime.now()
processed_documents = toma.simple.batch(run_batch, 100000, nlp, sents_tokenized)
# doc = nlp(sents_tokenized)
print(f"one go took: {datetime.now() - start}")

print(processed_documents[:1])

exit()

# sents = sents * 10
print("###################################")
print(len(sents))
for s in tqdm(sents):
    doc = nlp(s)  # run the pipeline


# CPU: 
# one go took: 0:00:02.009349
# ###################################
# 700
# 100%|███| 700/700 [01:08<00:00, 10.25it/s]

dh = get_dict("msd_en", refresh=True)

for s in sents:
    doc = nlp(s)  # run the pipeline
    # print(doc.to_dict())

    for sentence in doc.sentences:
        for word in sentence.words:
            # print(word)
            msd = dh[word.xpos]
            print(green(word.text) if msd["case"] == 2 else
                  red(word.text) if msd["case"] == 4 else bold(word.text))
            print("\t", word.lemma, word.pos, word.xpos, msd)

