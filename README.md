# EssayHelper

V tem repozitoriju je objavljen program, ki smo ga razvili v sklopu diplomske naloge na *Fakulteti za računalništvo in informatiko*.
Naslov diplosmkega dela je ***Predlogi jezikovnih popravkov v slovenščini z modelom SloBERTa***.
Spodaj so objavljena kratka navodila za postavitev okolja in zagon programa.


V repozitoriju smo objavili tudi tri datoteke, ki smo jih uporabili za učeneje mejnih vrednosti
in evalvacijo rešitve. Nahajajo se v direktoriju `corpus/Lektor/`.
Datoteke v vrsticah vsebujejo lektorirane in prečiščene povedi iz korpusa [Lektor](https://slovenscina.u3p.si/korpusi/lektor/).


## Primer izpisa obdelave besedila:

<img src="images/primer_konca_izpisa.png" width="65%">



## Quickstart and installation instructions
1. requirements:
    - conda
    - cuda drivers (optionally?)
2. Get the repo:

    `git clone https://github.com/timopetric/EssayHelper.git`

    `cd EssayHelper`

3. Create conda environment:

    `conda env create -f environment.yml`
    
    `conda activate EssayHelper`

4. Download required files (only once):

    `python src/download_files.py`

5. Run program `hello_world.py` or `SloHelper.py` for eval:

    `python src/hello_world.py`

#### Backup conda env:

`conda env export --no-builds > environment.yml`

#### Update conda env:

`conda env update -f environment.yml`





<br><br>
*Contact me if you have any questions or have truble setting up the environment*
