from google.cloud import translate_v2 as translate
import os
from typing import List, Tuple, Any
import pandas as pd


class Translator():
    def __init__(self) -> None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
        self.translate_client_ = translate.Client()

    def double_translation(self, target: str, text: str) -> str:
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        result_translation = self.translate_client_.translate(
            text, target_language=target)
        result = self.translate_client_.translate(
            result_translation["translatedText"],target_language='en-us')

        return result["translatedText"]



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


def load_dataframe(filename: str) -> pd.DataFrame:
    df = pd.DataFrame()
    try:
        df = pd.read_csv(filename)
    except:
        pass
    return df


def filter_copies_only(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['same'] == 1]
    df.reset_index(drop=True, inplace=True)
    return df


def generate_double_translations(df: pd.DataFrame,
                                 start_idx: int) -> pd.DataFrame:
    translator = Translator()
    translations: List[Tuple[Any]] = []
    for index, row in df.iterrows():
        if index < start_idx:
            continue
        print(index)
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        type_plagiarism = row['type_plagiarism']
        try:
            sentence1_translation: str = translator.double_translation("zh-cn", sentence1)
            sentence2_translation: str = translator.double_translation("zh-cn", sentence2)
            translations.append((sentence1, sentence1_translation, 1, type_plagiarism))
            translations.append((sentence1, sentence2_translation, 1, type_plagiarism))
            translations.append((sentence2, sentence1_translation, 1, type_plagiarism))
            translations.append((sentence2, sentence2_translation, 1, type_plagiarism))
        except:
            break
    
    return pd.DataFrame(translations, columns=df.columns)


def main() -> None:
    start_sentence_idx = load_start_sentence_idx()
    
    df: pd.DataFrame = load_dataframe("sentences.csv")
    df = filter_copies_only(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True, ignore_index=True)
    df = df.sample(frac=1, ignore_index=True, random_state=42)
    translations_df: pd.DataFrame = generate_double_translations(df, start_sentence_idx)
    print(translations_df)

    end_sentence_idx = len(translations_df) // 4 + start_sentence_idx - 1
    print(end_sentence_idx + 1)
    dump_end_sentence_idx(end_sentence_idx + 1)

    previous_df = load_dataframe("translation_sentences.csv")
    translations_df = pd.concat([previous_df, translations_df])
    print(translations_df.head())
    
    translations_df.to_csv("translation_sentences.csv", index = False)



if __name__=="__main__":
    main()