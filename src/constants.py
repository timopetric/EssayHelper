from pathlib import Path
from os.path import join
import os


repo_root = join(str(Path.home()), "EssayHelper")  # Change repo root if you changed the name

FOLDERS_THAT_HAVE_TO_EXIST = ["tmp", "dictionaries", "test_input_texts", "corpus"]
for folder_path in [join(repo_root, p) for p in FOLDERS_THAT_HAVE_TO_EXIST]:
    os.makedirs(folder_path, exist_ok=True)

PTH_TEST_INPUT_TEXTS_FOL = join(repo_root, "test_input_texts")

PTH_PKL_FILEPATH_MSD_SL = join(repo_root, "tmp/dict_msd_sl.pkl")
PTH_PKL_FILEPATH_MSD_EN = join(repo_root, "tmp/dict_msd_en.pkl")
PTH_PKL_FILEPATH_MSD_CATEGORIES= join(repo_root, "tmp/dict_msd_cats.pkl")
PTH_PKL_FILEPATH_SLOLEKS_FORMS = join(repo_root, "tmp/dict_sloleks_forms.pkl")
PTH_PKL_FILEPATH_SLOLEKS_PAIRS = join(repo_root, "tmp/dict_sloleks_pairs.pkl")

PTH_SHERPA_THRERSHOLDS = join(repo_root, "tmp/sherpa_thresholds")
PTH_SOLAR_CORPUS = join(repo_root, "corpus/Solar2.0-Error/solar2-error.xml")

URL_MDS_SL_SPC_XML = "https://raw.githubusercontent.com/clarinsi/mte-msd/059773708e953cef58f9b7119de0cb3aa92553bb/xml/msd-sl.spc.xml"
URL_SLOLEKS_2_0_MTE_ZIP = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1230/Sloleks2.0.MTE.zip"
URL_MSD_SPEC_CODES = "https://raw.githubusercontent.com/mgrcar/OPA/6f0b45920f562db5a32aa23ac258ab4c759c2a51/Obeliks/LemmatizerTrain/MsdSpecsSloSloCodes.txt"

MD5_HASH_MDS_SL_SPC_XML = "e9cf496f2b69b7ea863a281ba418b7ea"
MD5_HASH_SLOLEKS_2_0_MTE_UNZIPED_DIR = "d41d8cd98f00b204e9800998ecf8427e"
MD5_HASH_MSD_SPEC_CODES = "39c8d8c70f1269756ffb8a5e4a99d7c4"

PTH_DICT_MSD_SL_SPC = join(repo_root, "dictionaries/msd-sl.spc.xml")
PTH_DICT_MSD_SPC_SLO_CODES = join(repo_root, "dictionaries/MsdSpecsSloSloCodes.txt")
PTH_DICT_SLOLEX_2_SL_TBL = join(repo_root, "dictionaries/sloleks/Sloleks2.0.MTE/sloleks_clarin_2.0-sl.tbl")
PTH_DICT_SLOLEX_2_SL_TBL_EXTRACT_LOCATION = join(repo_root, "dictionaries/sloleks/")

PTH_PARAM_OPTIMIZATION_STUDIES = join(repo_root, "tmp/sherpa_thresholds/")

PTH_LEKTOR_OUT = join(repo_root, "corpus/Lektor/lektor_out.txt")
PTH_LEKTOR_OUT_TRAIN = join(repo_root, "corpus/Lektor/lektor_out_train.txt")
PTH_LEKTOR_OUT_TEST = join(repo_root, "corpus/Lektor/lektor_out_test.txt")

RE_DEFAULT_TOKEN_STR_FILTER = r"[A-Za-zčšćđžČĆŠĐŽ]{2,}"
RE_DEFAULT_TOKEN_STR_FILTER_1 = r"[A-Za-zčšćđžČĆŠĐŽ]{1,}"

SLOBERTA_MCD_MASK_TOKEN_STR = "<mask>"


