import re
import constant
from num2words import num2words
from utils import transcript_to_index


# Letter need to substitute
substitute_from = 'é:'
substitute_to = 'e '

# Punc need to remove
remove_punctuation = '!"(),/=?[].-“”'

# Hardcoded Remove labels and audio descriptions
label_remove = {"Mitsuha: ", "Taki: ", "Sayaka: ", "Yotsuha: ", "Taki's Father: ", 
         "From Tsukasa: ", "Diner: ", "JR Yamanote Line: ", "Mitsuha (as Taki): ",
         "Phone: ", "Okudera: ", "Takagi: ", "Waitress: ", "Chef: ", "Distant Voice: ", 
         "Taki (as Mitsuha): ", "Mitsuha's Father: ", "Train Announcer: ", "Train announcer: ",
         "Tsukasa: "}
label_re = "|".join(set(map(re.escape, label_remove)))

# Compile regex
label_pattern = re.compile(label_re)            # Hardcoded remove
space_pattern = re.compile(r"\s+")              # Combine multiple spaces into one
action_pattern = re.compile(r"\[.*\]")          # Remove action description like [gasp]
description_pattern = re.compile(r"\(.*\)")     # Remove descriptive notes like (20:30)
number_pattern = re.compile(r"[0-9]+")          # Convert number to English words


def preprocess_line(line):
    line = label_pattern.sub("", line.strip())
    line = action_pattern.sub("", line)
    line = description_pattern.sub("", line)
    line = line.translate(str.maketrans(substitute_from, substitute_to, remove_punctuation))
    line = number_pattern.sub(lambda match: num2words(match[0]).replace(',', ''), line)
    line = space_pattern.sub(" ", line)
    line = line.lower().strip()
    return line


def preprocess(sub_df, remove_music=False, min_label_cnt=constant.MIN_LABEL_CNT):
    # Process text
    sub_df['Processed Text'] = sub_df['Text'].apply(preprocess_line)

    # Remove band
    if remove_music:
        processed_df = sub_df[sub_df["Speaker Label"] != "Band"]
    else:
        processed_df = sub_df

    # Keep only valid lines
    processed_df = processed_df[(processed_df['Processed Text'] != '') & (processed_df['Speaker Label'] != 'None')].copy()
    print("Number of lines:", len(sub_df), '->', len(processed_df))
    processed_df['Transcript Indices'] = processed_df['Processed Text'].apply(transcript_to_index)

    # Deal with speaker label
    processed_df = processed_df.reset_index()
    valid_labels = processed_df.groupby("Speaker Label").index.count() > min_label_cnt

    # Keep only labels that appear more than 5 times
    mask = valid_labels[processed_df["Speaker Label"]].reset_index()
    processed_df["Speaker Label"].where(mask['index'], 'None', inplace=True)
    constant.set_min_label_count(min_label_cnt)
    label_set = set(processed_df['Speaker Label'])
    assert len(constant.LABEL_LIST) - len(label_set) <= 1
    print("Number of labels:", len(label_set))
    return processed_df


# Check letter list after proc
def get_letter_vocab(series):
    letters = set()
    for line in series:
        letters |= set(line)
    return letters
