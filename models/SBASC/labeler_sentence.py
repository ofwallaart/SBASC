import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from models.SBASC.config import *
import re
from sklearn.metrics import classification_report


def load_training_data(file_path):
    sentences = []
    for line in open(file_path, encoding="utf-8"):
        split_lines = list(filter(None, re.split('; |\. |\! |\n| \?', line.lower())))
        for split_line in split_lines:
            sentences.append(split_line.strip())
    return sentences


def load_evaluate_data(path):
    test_sentences = []
    test_cats = []
    test_pols = []

    with open(f'{path}/test.txt', 'r', encoding="utf8") as f:
        for line in f:
            _, cat, pol, sentence = line.strip().split('\t')
            cat = int(cat)
            pol = int(pol)
            test_cats.append(cat)
            test_pols.append(pol)
            test_sentences.append(sentence)
    return test_sentences, test_cats, test_pols


class Labeler:
    def __init__(self):
        self.domain = config['domain']
        self.model = SentenceTransformer(sbert_mapper[self.domain], device=config['device'])
        self.cat_threshold = 0.7
        self.pol_threshold = 0.5
        self.root_path = path_mapper[self.domain]
        self.categories = aspect_category_mapper[self.domain]
        self.polarities = sentiment_category_mapper[self.domain]
        self.labels = None
        self.sentences = None

    def __call__(self, evaluate=True, load=False):
        category_seeds = aspect_seed_sentence_mapper[self.domain]
        polarity_seeds = sentiment_seed_sentence_mapper[self.domain]

        split = [len(self.categories), len(self.polarities)]

        # Seeds
        seeds = {}
        seeds_len = []
        for cat in self.categories:
            seeds[cat] = list(category_seeds[cat])
            seeds_len.append(len(category_seeds[cat]) + 1)
        for pol in self.polarities:
            seeds[pol] = list(polarity_seeds[pol])
            seeds_len.append(len(polarity_seeds[pol]) + 1)

        seed_embeddings = [self.model.encode(seed, convert_to_tensor=True) for seed in list(seeds.values())]

        # Load and encode the train set
        self.sentences = load_training_data(f'{self.root_path}/train.txt')

        if load:
            print(f'Loading embeddings from {self.root_path}')
            embeddings = torch.load(f'{self.root_path}/sbert_train_embeddings.pickle')
        else:
            embeddings = self.model.encode(self.sentences, convert_to_tensor=True, show_progress_bar=True)
            print(f'Saving embeddings to {self.root_path}')
            torch.save(embeddings, f'{self.root_path}/sbert_train_embeddings.pickle')

        cosine_scores = []
        for seed_embedding in seed_embeddings:
            # Compute cosine-similarities
            total_tensor = torch.cat((seed_embedding, torch.mean(seed_embedding, dim=0).unsqueeze(0)))
            cosine_scores.append(torch.max(util.cos_sim(total_tensor, embeddings), dim=0)[0].unsqueeze(dim=-1))

        cosine_category_scores, cosine_polarity_scores = torch.split(torch.cat(cosine_scores, 1), split, -1)

        category_max, category_argmax = cosine_category_scores.max(dim=-1)
        polarity_max, polarity_argmax = cosine_polarity_scores.max(dim=-1)

        labels = np.array(
            [category_argmax.tolist(), category_max.tolist(), polarity_argmax.tolist(), polarity_max.tolist(), np.arange(0, len(self.sentences))])

        self.labels = labels

        # df = pd.DataFrame(labels.transpose())
        # df = df.groupby([0, 2]).agg(percentile_cat = (1,lambda x: x.quantile(0.9)), percentile_pol = (3,lambda x: x.quantile(0.9))).reset_index().to_numpy()
        #
        # def select_percentiles(x):
        #     return df[np.where((df[:, 0]==x[0]) & (df[:, 1]==x[2])), 2]
        #
        # def select_percentiles_pol(x):
        #     return df[np.where((df[:, 0]==x[0]) & (df[:, 1]==x[2])), 3]
        #
        # labels = np.r_[labels, np.apply_along_axis(select_percentiles, axis=0, arr=labels).squeeze((0)), np.apply_along_axis(select_percentiles_pol, axis=0, arr=labels).squeeze((0))]
        #
        # # No conflict (avoid multi-class sentences)
        # labels = np.transpose(labels[:, ((labels[1, :] >= labels[5, :])) & ((labels[3, :] >= labels[6, :]))])

        labels = np.transpose(labels[:, (labels[1, :] >= self.cat_threshold) & (labels[3, :] >= self.pol_threshold)])

        nf = open(f'{self.root_path}/label-sentences.txt', 'w', encoding="utf8")
        cnt = {}

        for label in labels:
            sentence = self.sentences[int(label[4])]
            aspect = self.categories[int(label[0])]
            sentiment = self.polarities[int(label[2])]
            nf.write(f'{sentence}\n')
            nf.write(f'{aspect} {sentiment}\n')
            keyword = f'{aspect}-{sentiment}'
            cnt[keyword] = cnt.get(keyword, 0) + 1

        nf.close
        # Labeled data statistics
        print('Labeled data statistics:')
        print(cnt)

        if evaluate:
            test_sentences, test_cats, test_pols = load_evaluate_data(self.root_path)
            test_embeddings = self.model.encode(test_sentences, convert_to_tensor=True, show_progress_bar=True)

            cosine_test_scores = []
            for seed_embedding in seed_embeddings:
                # Compute cosine-similarities
                total_tensor = torch.cat((seed_embedding, torch.mean(seed_embedding, dim=0).unsqueeze(0)))
                cosine_test_scores.append(torch.max(util.cos_sim(total_tensor, test_embeddings), dim=0)[0].unsqueeze(dim=-1))

            cosine_category_test_scores, cosine_polarity_test_scores = torch.split(torch.cat(cosine_test_scores, 1), split, -1)

            category_test_argmax = cosine_category_test_scores.argmax(dim=-1).tolist()
            polarity_test_argmax = cosine_polarity_test_scores.argmax(dim=-1).tolist()

            print("Polarity")
            print(classification_report(
                test_pols, polarity_test_argmax, target_names=sentiment_category_mapper[self.domain], digits=4
            ))
            print()

            print("Aspect")
            print(classification_report(
                test_cats, category_test_argmax, target_names=aspect_category_mapper[self.domain], digits=4
            ))
            print()
