import spacy
# sudo -H ~/virtualenvs/localml/bin/python3.10 -m spacy download en_core_web_lg

# exit code 138: segmentation default occurs
nlp = spacy.load("en_core_web_lg")

