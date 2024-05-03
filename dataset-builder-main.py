import nltk
from nltk.corpus import wordnet
import numpy as np
import pandas as pd
from copy import copy

from typing import Dict, Set, List, Tuple, Any


def read_files() -> List[str]:
    n = 110
    files_content: List[str] = []
    for i in range(1, n + 1):
        file_path = "data/"
        filename_prefix = "org-"
        filename_suffix = ".txt"
        padded_num_str = str(i).rjust(3, '0')
        full_filename = (file_path + filename_prefix + padded_num_str
                         + filename_suffix)
        file = open(full_filename, "r")
        content = file.read()
        clean_content = content.strip()
        files_content.append(clean_content)
        file.close()
    return files_content


def split_sentences(files_content: List[str]) -> List[str]:
    sentences: List[str] = []
    for paragraph in files_content:
        sentences += paragraph.split(".")
    return sentences


def make_different_pairs(sentences: List[str]) -> List[Tuple[str]]:
    pairs: List[Tuple[str]] = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            pairs.append((sentences[i], sentences[j]))
    return pairs


def make_duplicate_pairs(sentences: List[str]) -> List[Tuple[str]]:
    pairs: List[Tuple[str]] = []
    for sentence in sentences:
        pairs.append((sentence, sentence))
    return pairs


def make_synonym_pairs(sentences: List[str]) -> List[Tuple[str]]:
    pairs: List[Tuple[str]] = []
    for sentence in sentences:
        tokens: list[str] = nltk.word_tokenize(sentences[0])
        tagged: list[Tuple[Any, str]] = nltk.pos_tag(tokens)
        words_synonyms: Dict[str, List] = dict()
        max_synonyms_any_word = 0
        for word, tag in tagged:
            if tag.startswith("NN"):
                # Ignore acronyms
                if (word.isupper()):
                    continue

                synonyms: Set[str] = set()
                for synonym in wordnet.synsets(word, pos=wordnet.NOUN):
                    for lemma in synonym.lemmas():
                        synonyms.add(lemma.name().replace("_", " "))
                words_synonyms[word] = list(synonyms)
                max_synonyms_any_word = max(max_synonyms_any_word,
                                            len(synonyms))

        for i in range(max_synonyms_any_word):
            changed_sentence = copy(sentence)
            for word in words_synonyms.keys():
                synonyms = words_synonyms[word]
                changed_sentence = changed_sentence.replace(
                    word, synonyms[min(len(synonyms) - 1, i)])
            pairs.append((sentence, changed_sentence))
        
    return pairs


def make_dataframe(different_sentences: List[Tuple[str]],
                   same_sentences: List[Tuple[str]]) -> pd.DataFrame:
    df_different: pd.DataFrame = pd.DataFrame(different_sentences,
                                              columns=["sentence1",
                                                       "sentence2"])
    df_different["same"] = 0
    df_different["type_plagiarism"] = "different"
    df_same: pd.DataFrame = pd.DataFrame(same_sentences,
                                         columns=["sentence1",
                                                  "sentence2"])
    df_same["same"] = 1
    df_same["type_plagiarism"] = "copy"
    return pd.concat([df_different, df_same])


def load_dataframe(csv_filename) -> pd.DataFrame:
    df = pd.DataFrame()
    try:
        df = pd.read_csv(csv_filename)
    except:
        pass
    return df


def main() -> None:
    files_content: List[str] = read_files()
    sentences: List[str] = split_sentences(files_content)
    different_pairs: List[Tuple[str]] = make_different_pairs(sentences)
    duplicate_pairs: List[Tuple[str]] = make_duplicate_pairs(sentences)
    synonym_pairs = make_synonym_pairs(sentences)
    df: pd.DataFrame = make_dataframe(different_pairs, synonym_pairs + duplicate_pairs)
    df_passive: pd.DataFrame = load_dataframe("passive_sentences.csv")
    df_double_negation: pd.DataFrame = load_dataframe("double_negation_sentences.csv")
    df_future_tense: pd.DataFrame = load_dataframe("future_tense_sentences.csv")
    df_paraphrase: pd.DataFrame = load_dataframe("paraphrase_sentences.csv")
    df_translation: pd.DataFrame = load_dataframe("translation_sentences.csv")
    df = pd.concat([df, df_passive, df_double_negation,
                    df_future_tense, df_paraphrase, df_translation])
    print(df.describe())
    df.to_csv("sentences.csv", index = False)

if __name__=="__main__":
    main()