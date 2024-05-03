import google.generativeai as genai

from typing import List, Tuple

from dotenv import load_dotenv
import os
import pandas as pd
import sys


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


def clean_response(gemini_response: str) -> str:
    token = "Change of tense:"
    idx = gemini_response.rfind(token)
    if idx != -1:
        return gemini_response[idx+len(token) + 1:]
    else:
        return gemini_response


def make_passive_sentences(sentences: List[str], 
                           start_sentence_idx: int) -> List[Tuple[str]]:
    pairs: List[Tuple[str]] = []
    load_dotenv()

    GEMINI_KEY = os.getenv('GEMINI_KEY')

    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-pro')
    for i in range(start_sentence_idx, len(sentences)):
        print(i)
        try:

            response = model.generate_content(
                """
                Original: This study provided a content analysis of studies aiming to
                disclose how artificial intelligence (AI) has been applied to the education
                sector and explore the potential research trends and challenges of AI
                in education.
                Change of tense: This study has provided a content analysis of studies
                that have aimed to disclose how artificial intelligence (AI) has been applied to the education
                sector and have explored the potential research trends and challenges of AI
                in education.
                Original: {\\sentence}
                Change of tense: 
                """.replace("{\\sentence}", sentences[i])
                )
            cleaned = clean_response(response.text)
            pairs.append((sentences[i], cleaned))
        except:
            break
    
    return pairs
        



def make_dataframe(same_sentences: List[Tuple[str]]) -> pd.DataFrame:
    df_same: pd.DataFrame = pd.DataFrame(same_sentences,
                                         columns=["sentence1",
                                                  "sentence2"])
    df_same["same"] = 1
    df_same["type_plagiarism"] = "tense"
    return df_same


def load_start_sentence_idx() -> int:
    start_sentence_idx = 0
    try:
        file = open("start_sentence_idx.txt", "r")
        start_sentence_idx = int(file.read())
        file.close()
    except:
        pass
    return start_sentence_idx


def dump_end_sentence_idx(end_sentece_idx: int) -> None:
    try:
        with open("start_sentence_idx.txt", "w") as file:
            file.write(str(end_sentece_idx))
    except:
        print(f"id not saved, id={end_sentece_idx}")


def main() -> None:
    start_sentence_idx = load_start_sentence_idx()
    files_content: List[str] = read_files()
    sentences: List[str] = split_sentences(files_content)
    passive_sentences: List[str] = make_passive_sentences(
        sentences, start_sentence_idx)
    
    end_sentence_idx = len(passive_sentences) + start_sentence_idx - 1
    dump_end_sentence_idx(end_sentence_idx + 1)
    df: pd.DataFrame = make_dataframe(passive_sentences)
    print(df.head())
    previous_df = pd.DataFrame()
    try:
        previous_df = pd.read_csv("future_tense_sentences.csv")
        print(previous_df.head())
        df = pd.concat([previous_df, df])
        print(df.head())
    except:
        pass
    
    df.to_csv("future_tense_sentences.csv", index = False)

if __name__=="__main__":
    main()