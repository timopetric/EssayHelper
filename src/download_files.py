import requests, zipfile, io
from constants import *
from typing import List
from pathlib import Path
from filehash import FileHash
from helpers import Location


def md5(path, md5hasher):
    assert isinstance(path, Path)
    if path.is_dir():
        return md5hasher.cathash_dir(path)
    elif path.is_file():
        return md5hasher.hash_file(path)
    elif not path.exists():
        return ""
    else:
        raise ValueError("path not folder nor file.")


def dl_extract(url: str, path: str, extract: bool):
    try:
        r = requests.get(url, allow_redirects=True)
    except requests.exceptions.ConnectionError as e:
        raise ValueError("Site seems to be unaccessible.")
        
    if extract:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path)
    else:
        with open(path, "wb") as f:
            f.write(r.content)


def check_files():
    locations: List[Location] = []

    # http://nl.ijs.si/ME/Vault/V5/msd/html/msd-sl.html#msd.msds-sl
    locations.append(Location(
        url=URL_MDS_SL_SPC_XML,
        path=PTH_DICT_MSD_SL_SPC,
        extract=False,
        md5=MD5_HASH_MDS_SL_SPC_XML))

    # sloleks2.0 https://www.clarin.si/repository/xmlui/handle/11356/1230
    locations.append(Location(
        url=URL_SLOLEKS_2_0_MTE_ZIP,
        path=PTH_DICT_SLOLEX_2_SL_TBL_EXTRACT_LOCATION,
        extract=True,
        md5=MD5_HASH_SLOLEKS_2_0_MTE_UNZIPED_DIR))

    locations.append(Location(
        url=URL_MSD_SPEC_CODES,
        path=PTH_DICT_MSD_SPC_SLO_CODES,
        extract=False,
        md5=MD5_HASH_MSD_SPEC_CODES))
    
    md5hasher = FileHash('md5')
    for location in locations:
        try:
            path = Path(location.path)
            hash_str = md5(path, md5hasher)

            if hash_str == location.md5:
                print(f"OK  {path}")
                continue
            print(f"Path: {path} has md5 hash: {hash_str} that does not match the hardcoded one.")

            # make folders
            path.parents[0].mkdir(parents=True, exist_ok=True)

            dl_extract(location.url, path, location.extract)
        except Exception as e:
            raise Exception("Something went wrong when checking files.")





def main():
    check_files()
    
if __name__ == "__main__":
    main()
