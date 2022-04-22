import os
import xml.etree.ElementTree as ET
import nltk
import pandas as pd
import random
import io
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download('punkt')
tqdm.pandas()


def read_data_2016(file, domain):
    """
    Reads data from SemEval-2016 XML format, filters relevant sentences for test set and creates a dataframe containing
    all test sentences
    :param file: the xml file from SemEval to be read
    :param domain: specified domain e.g. REST, LAPT
    :return:dataframe containing all test sentences
    """
    if os.path.isfile(file) == False:
        raise ("[!] Data %s not found" % file)

    # parse xml file to tree
    tree = ET.parse(file)
    root = tree.getroot()

    df_dict = {
        'sentence_index': [],
        'sentence': [],
        'sent_len': [],
        'opinion_num': [],
        'opinion_count': [],
        'polarity': [],
        'target': [],
        'aspect': [],
        'imp': []
    }

    for index, sentence in enumerate(root.iter('sentence')):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)

        sent = " ".join(word_tokenize(sent.lower(), language='dutch'))

        sent_len = len(sptoks)

        for opinions in sentence.iter('Opinions'):
            for i, opinion in enumerate(opinions.findall('Opinion')):
                df_dict['sentence_index'].append(index)
                df_dict['opinion_count'].append(len(opinions.findall('Opinion')))
                df_dict['sentence'].append(sent)
                df_dict['sent_len'].append(sent_len)

                df_dict['opinion_num'].append(i + 1)
                df_dict['polarity'].append(opinion.get("polarity"))
                df_dict['target'].append(opinion.get('target'))
                df_dict['aspect'].append(opinion.get('category'))
                if opinion.get('target') != 'NULL':
                    df_dict['imp'].append(True)
                else:
                    df_dict['imp'].append(False)

    df = pd.DataFrame.from_dict(df_dict)
    df[['aspect_group', 'aspect_subgroup']] = df['aspect'].str.split('#', expand=True)

    # Domain specific filtering
    if domain == "REST":
        df = df[df.aspect_group != "RESTAURANT"]
    elif domain == "LAPT":
        value_list = ['SUPPORT', 'OS', 'DISPLAY', 'BATTERY', 'COMPANY', 'MOUSE', 'SOFTWARE', 'KEYBOARD']
        df = df[df.aspect_group.isin(value_list)]
    elif domain == "PHNS":
        value_list = ['MULTIMEDIA_DEVICES', 'DISPLAY', 'BATTERY', 'SOFTWARE']
        df = df[df.aspect_group.isin(value_list)]

    # Filter opinion_count == 1 and polarity != 'neutral'
    df['new_count'] = df.groupby(['sentence_index'])['sentence_index'].transform("count")
    df['uniform_count'] = df.groupby(['sentence_index', 'aspect_group'])['sentence_index'].transform("count")
    df = df[(df.new_count == df.opinion_count) & (df.new_count == df.uniform_count) & (df.polarity != 'neutral')]
    df = df[(df.opinion_num == 1)]

    df_polarity_count = df.groupby(['polarity']).count()
    df_polarity_count['pct'] = df_polarity_count.sentence * 100 / df_polarity_count.sentence.sum()
    df_polarity_count = df_polarity_count[['pct', 'sentence']]

    df_aspect_count = df.groupby(['aspect_group']).count()
    df_aspect_count['pct'] = df_aspect_count.sentence * 100 / df_aspect_count.sentence.sum()
    df_aspect_count = df_aspect_count[['pct', 'sentence']]

    df['polarity'].replace(to_replace=['positive', 'negative'], value=[1, 0], inplace=True)
    df['aspect_group'].replace(to_replace=['LOCATION', 'DRINKS', 'FOOD', 'AMBIENCE', 'SERVICE'], value=[0, 1, 2, 3, 4], inplace=True)

    print(df.shape)

    return df_polarity_count, df_aspect_count, df.reset_index(drop=True)


def write_test_text(fn, df):
    """

    :param fn:
    :param df:
    """
    with io.open(fn, "w", encoding='utf8') as fw:
        for index, row in df.iterrows():
            fw.write(f"{index}\t{row.aspect_group}\t{row.polarity}\t{row.sentence}\n")


def load_dutch_restaurant(path):
    """
    Loads Dutch restaurant review dataset and process sentences for creating a DK-BERT model and a train set for SBASC.
    :param path: the folder where domain_train and domain_dev files are stored.
    """
    REVIEWS = (
        "https://bhciaaablob.blob.core.windows.net/cmotionsnlpblogs/RestoReviewRawdata.csv"
    )
    resto = pd.read_csv(REVIEWS, decimal=",")

    # fix utf-8 encoding
    resto.reviewText = resto.reviewText.fillna("b''").apply(
        lambda b: eval(b + ".decode('utf-8')")
    )

    resto.reviewText = resto.reviewText.progress_apply(lambda x: " ".join(word_tokenize(str(x).lower(), language='dutch')) )


    def write_text(fn, data, skip_tag=True):
        with io.open(fn, "w", encoding='utf8') as fw:
            for (tag, rating, text) in data:
                if not skip_tag:
                    fw.write("\n" + tag + " " + rating + "\n")
                fw.write(text + "\n")


    keep_probs = 0.8

    train_data = []
    test_data = []
    fields = ['reviewText']  # ,'ToelichtingDetractors', 'ToelichtingPassives', 'ToelichtingPromoters']

    for index, row in resto.iterrows():
        if len(row['reviewText']) > 0:
            rnd = random.random()
            if rnd < keep_probs:
                if rnd < 0.0005:
                    test_data.append((row['restoId'], row['reviewScoreOverall'], row['reviewText']))
                else:
                    train_data.append((row['restoId'], row['reviewScoreOverall'], row['reviewText']))
            else:
                test_data.append((row['restoId'], row['reviewScoreOverall'], row['reviewText']))

    split_test_as_dev2 = int(len(test_data) * 0.05)
    write_text(f'{path}/domain_train.txt', train_data)
    write_text(f"{path}/domain_dev.txt", test_data[:split_test_as_dev2])
    write_text(f"{path}/train.txt", test_data[split_test_as_dev2:30000])

    # Create the test set
    nl_rest = f"{path}/DU_REST_SB1_TEST.xml"
    _, _, nl_df = read_data_2016(nl_rest, 'REST')
    write_test_text(f"{path}/test.txt", nl_df)


if __name__ == '__main__':
    load_dutch_restaurant("data/restaurant-nl")
