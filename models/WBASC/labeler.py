import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from models.WBASC.config import *
import re
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

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


def get_rep_sentences(self, embeddings, cosine_scores_train, aspect_seed, aspect_category, embeddings_marco, N,
                      test_embeddings=None):
    train_sentences = self.sentences
    topk_scores = []
    topk = torch.topk(cosine_scores_train, cosine_scores_train.shape[1], dim=1).indices

    # If desired also add individual seed word matches to matchable sentences
    topk_scores_indiv = []
    seeds_indiv = [item for sublist in list(aspect_seed.values()) for item in sublist]
    seeds_len = [len(i) for i in aspect_seed.values()]
    seeds_indiv_embeddings = self.marco_model.encode(seeds_indiv, convert_to_tensor=True, show_progress_bar=True)
    train_embeddings_shortened = [torch.unsqueeze(embeddings_marco[i], -1) for i, x in enumerate(train_sentences) if
                                   len(x.split()) >= 4]
    cosine_scores_indiv_train = torch.topk(
        util.cos_sim(seeds_indiv_embeddings, torch.cat(train_embeddings_shortened, -1).T), 1, dim=1)

    for i, argmax_cosine_score in enumerate(torch.split(cosine_scores_indiv_train.indices, seeds_len)):
        topk_scores_indiv.append(
            torch.index_select(torch.cat(train_embeddings_shortened, -1).T, 0, argmax_cosine_score.squeeze(-1)))

    for idx, top in enumerate(topk):

        # Make sure sentences contain at least one seed word and no seed words from other aspects
        seeds_in_sent = []
        seeds_not_in_sent = []
        for i in top.tolist():
            seeds_in_sent.append(
                [ele for ele in list(aspect_seed[aspect_category[idx]]) if
                 (" " + ele + " " in train_sentences[i])])

            lists = []
            for not_list_item in aspect_category:
                if not_list_item != aspect_category[idx]:
                    lists.extend(list(aspect_seed[not_list_item]))
            seeds_not_in_sent.append([ele for ele in lists if (ele in train_sentences[i])])

        # Do the checks
        final_top = []

        # Try to make at least each seed word appear in a sentence
        for seed in aspect_seed[aspect_category[idx]]:
            g = [i for i, e in enumerate(seeds_in_sent) if seed in e]
            for g_item in g:
                sentence_index = top.tolist()[g_item]
                if seeds_not_in_sent[g_item] \
                        or sentence_index in final_top \
                        or len(train_sentences[sentence_index].split()) <= 1 \
                        or train_sentences[sentence_index] in [train_sentences[i] for i in final_top]:
                    continue
                else:
                    final_top.append(sentence_index)
                    break

        # Fill top sentences to size N with most relevant sentences
        for i, t_item in enumerate(top.tolist()):
            if seeds_not_in_sent[i] or t_item in final_top or len(train_sentences[t_item].split()) <= 1:
                continue
            else:
                if len(final_top) > N[self.domain]:
                    break
                final_top.append(t_item)

        sent_embeddings = torch.stack([embeddings[i] for i in final_top])

        # Also include the average of top K sentences
        topk_embeddings = torch.cat(
            (sent_embeddings, torch.mean(sent_embeddings, dim=0).unsqueeze(0), topk_scores_indiv[idx]))
        # topk_embeddings = torch.cat((sent_embeddings, torch.mean(sent_embeddings, dim=0).unsqueeze(0)))
        # topk_embeddings = torch.mean(sent_embeddings, dim=0).unsqueeze(0)
        # topk_embeddings = sent_embeddings

        # Compute cosine-similarities between top representing sentences and all other train/test data
        if torch.is_tensor(test_embeddings):
            topk_scores.append(torch.max(util.cos_sim(topk_embeddings, test_embeddings), dim=0)[0].unsqueeze(dim=-1))
        else:
            topk_scores.append(torch.max(util.cos_sim(topk_embeddings, embeddings), dim=0)[0].unsqueeze(dim=-1))

            for i in final_top:
                print(train_sentences[i])
            print()

    return torch.t(torch.cat(topk_scores, 1))


class Labeler:
    def __init__(self, cfg):
        self.domain = cfg.domain.name
        self.model = SentenceTransformer(cfg.domain.sbert_mapper, device=cfg.device)
        self.marco_model = SentenceTransformer('msmarco-distilbert-base-v4', device=config['device'])
        self.cat_threshold = cfg.domain.cat_threshold
        self.pol_threshold = cfg.domain.pol_threshold
        self.root_path = cfg.domain.path_mapper
        self.categories = cfg.domain.aspect_category_mapper
        self.polarities = cfg.domain.sentiment_category_mapper
        self.category_sentences = cfg.domain.aspect_seed_mapper
        self.sentiment_sentences = cfg.domain.sentiment_seed_mapper
        self.N = cfg.model.N
        self.labels = None
        self.sentences = None

    def __call__(self, use_two_step=True, evaluate=True, load=False):

        category_seeds = self.category_sentences
        polarity_seeds = self.sentiment_sentences

        split = [len(self.categories), len(self.polarities)]

        # Seeds
        seeds = {}
        for cat in self.categories:
            seeds[cat] = " ".join(category_seeds[cat])
        for pol in self.polarities:
            seeds[pol] = " ".join(polarity_seeds[pol])

        seed_embeddings = self.model.encode(list(seeds.values()), convert_to_tensor=True, show_progress_bar=True)
        seed_embeddings_marco = self.marco_model.encode(list(seeds.values()), convert_to_tensor=True,
                                                        show_progress_bar=True)

        # Load and encode the train set
        self.sentences = load_training_data(f'{self.root_path}/train.txt')

        if load:
            print(f'Loading embeddings from {self.root_path}')
            embeddings = torch.load(f'{self.root_path}/sbert_train_embeddings.pickle')[:20000]
            embeddings_marco = torch.load(f'{self.root_path}/sbert_train_embeddings_marco.pickle')[:20000]
            self.sentences = self.sentences[:20000]
        else:
            embeddings = self.model.encode(self.sentences, convert_to_tensor=True, show_progress_bar=True)
            embeddings_marco = self.marco_model.encode(self.sentences, convert_to_tensor=True, show_progress_bar=True)
            print(f'Saving embeddings to {self.root_path}')
            torch.save(embeddings, f'{self.root_path}/sbert_train_embeddings.pickle')
            torch.save(embeddings_marco, f'{self.root_path}/sbert_train_embeddings_marco.pickle')

        # Compute cosine-similarities
        cosine_category_scores, cosine_polarity_scores = torch.split(
            util.cos_sim(seed_embeddings_marco, embeddings_marco), split)

        if use_two_step:
            # Get top most representable sentences for each aspect
            cosine_category_scores = get_rep_sentences(self, embeddings, cosine_category_scores,
                                                       category_seeds, self.categories, embeddings_marco, self.N)
            cosine_polarity_scores = get_rep_sentences(self, embeddings, cosine_polarity_scores,
                                                       polarity_seeds, self.polarities, embeddings_marco, self.N)

        category_argmax = torch.argmax(cosine_category_scores, dim=0).tolist()
        category_max = torch.max(cosine_category_scores, dim=0)[0].tolist()

        polarity_argmax = torch.argmax(cosine_polarity_scores, dim=0).tolist()
        polarity_max = torch.max(cosine_polarity_scores, dim=0)[0].tolist()

        labels = np.array(
            [category_argmax, category_max, polarity_argmax, polarity_max, np.arange(0, len(self.sentences))])

        self.labels = labels

        # No conflict (avoid multi-class sentences)
        labels = np.transpose(labels[:, (labels[1, :] >= self.cat_threshold) & (labels[3, :] >= self.pol_threshold)])

        nf = open(f'{self.root_path}/label-sbert.txt', 'w', encoding="utf8")
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

            if use_two_step:
                cosine_category_test_scores = get_rep_sentences(self, embeddings, cosine_category_scores,
                                                                category_seeds, self.categories, embeddings_marco,
                                                                test_embeddings, self.N)
                cosine_polarity_test_scores = get_rep_sentences(self, embeddings, cosine_polarity_scores,
                                                                polarity_seeds, self.polarities, embeddings_marco,
                                                                test_embeddings, self.N)
            else:
                cosine_category_test_scores, cosine_polarity_test_scores = torch.split(
                    util.cos_sim(seed_embeddings, test_embeddings), split)

            category_test_argmax = torch.argmax(cosine_category_test_scores, dim=0).tolist()
            polarity_test_argmax = torch.argmax(cosine_polarity_test_scores, dim=0).tolist()

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
            print(confusion_matrix(test_cats, category_test_argmax))

            return pd.DataFrame(data={'sentence': test_sentences, 'test_cats': test_cats, 'pred_cats': category_test_argmax, 'test_pols':test_pols, 'pred_pols':polarity_test_argmax}).sort_values(['pred_cats','test_cats'])

    def update_labels(self, cat_threshold, pol_threshold):
        labels = self.labels
        sentences = self.sentences

        # No conflict (avoid multi-class sentences)
        labels = np.transpose(labels[:, (labels[1, :] >= cat_threshold) & (labels[3, :] >= pol_threshold)])

        nf = open(f'{self.root_path}/label-sbert.txt', 'w', encoding="utf8")
        cnt = {}

        for label in labels:
            sentence = sentences[int(label[4])]
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


if __name__ == '__main__':
    torch._set_deterministic(True)

    labeler = Labeler()
    df = labeler(load=True)
