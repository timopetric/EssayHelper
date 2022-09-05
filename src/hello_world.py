from EssayHelper import EssayHelper

def main():
    test_input_text = """
    Naredil je dve čudne napake.
    Hotel je preizkusiti dve testne vožnje z avtom.
    Opazil je, da mu mankjata dve iztočnice.
    Včeraj nisem videl Petro.
    Že dolgo nisem jedel tako dobro solato.   
    Ne morem jo videti!
    """

    EssayHelper(test_input_text, cuda_device=0)


if __name__ == "__main__":
    main()
