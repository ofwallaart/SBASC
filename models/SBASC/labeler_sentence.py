import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, BertModel
from tqdm import trange
import os

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
    def __init__(self, cfg):
        self.cfg = cfg
        self.domain = cfg.domain.name
        self.model = SentenceTransformer(cfg.domain.sbert_mapper, device=cfg.device)
        self.cat_threshold = cfg.domain.cat_threshold
        self.pol_threshold = cfg.domain.pol_threshold
        self.root_path = cfg.path_mapper
        self.categories = cfg.domain.aspect_category_mapper
        self.polarities = cfg.domain.sentiment_category_mapper
        self.category_sentences = cfg.domain.aspect_seed_mapper
        self.sentiment_sentences = cfg.domain.sentiment_seed_mapper
        self.labels = None
        self.sentences = None

    def __call__(self, evaluate=True, load=False):
        category_seeds = self.category_sentences
        polarity_seeds = self.sentiment_sentences

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

        # Load and encode the seeds and train set
        self.sentences = load_training_data(f'{self.root_path}/train.txt')

        # Load different embeddings for ablation study
        if self.cfg.ablation == 'WithoutSBERT':
            seed_embeddings, embeddings = self.__bert_embedder(load, seeds)
        else:
            seed_embeddings, embeddings = self.__sbert_embedder(load, seeds)

        # Start the similarity finding process
        cosine_scores = []
        for seed_embedding in seed_embeddings:
            # Compute cosine-similarities
            total_tensor = torch.cat((seed_embedding, torch.mean(seed_embedding, dim=0).unsqueeze(0)))
            cosine_scores.append(torch.max(util.cos_sim(total_tensor, embeddings), dim=0)[0].unsqueeze(dim=-1))

        cosine_category_scores, cosine_polarity_scores = torch.split(torch.cat(cosine_scores, 1), split, -1)

        category_max, category_argmax = cosine_category_scores.max(dim=-1)
        polarity_max, polarity_argmax = cosine_polarity_scores.max(dim=-1)

        labels = np.array(
            [category_argmax.tolist(), category_max.tolist(), polarity_argmax.tolist(), polarity_max.tolist(),
             np.arange(0, len(self.sentences))])

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
                cosine_test_scores.append(
                    torch.max(util.cos_sim(total_tensor, test_embeddings), dim=0)[0].unsqueeze(dim=-1))

            cosine_category_test_scores, cosine_polarity_test_scores = torch.split(torch.cat(cosine_test_scores, 1),
                                                                                   split, -1)

            category_test_argmax = cosine_category_test_scores.argmax(dim=-1).tolist()
            polarity_test_argmax = cosine_polarity_test_scores.argmax(dim=-1).tolist()

            print("Polarity")
            print(classification_report(
                test_pols, polarity_test_argmax, target_names=self.polarities, digits=4
            ))
            print()

            print("Aspect")
            print(classification_report(
                test_cats, category_test_argmax, target_names=self.categories, digits=4
            ))
            print()

            return classification_report(test_pols, polarity_test_argmax, target_names=self.polarities, output_dict=True), classification_report(
                test_cats, category_test_argmax, target_names=self.categories, digits=6, output_dict=True), polarity_test_argmax, category_test_argmax

    def __bert_embedder(self, load, seeds):
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.domain.bert_mapper)
        model = BertModel.from_pretrained(self.cfg.domain.bert_mapper, output_hidden_states=True).to('cuda')

        batch_size = 24

        def embedder(sentence_list):
            sentence_embeddings = []

            for i in trange(0, len(sentence_list), batch_size):
                encoded_dict = tokenizer(
                    sentence_list[i:i+batch_size],
                    padding=True,
                    return_tensors='pt',
                    max_length=128,
                    return_attention_mask=True,
                    truncation=True)
                model.eval()
                with torch.no_grad():
                    outputs = model(encoded_dict['input_ids'].to('cuda'), encoded_dict['token_type_ids'].to('cuda'),
                                    encoded_dict['attention_mask'].to('cuda'))
                x = outputs[2][11]
                mask = encoded_dict['attention_mask'].to('cuda')  # (bsz, seq_len)
                se = x * mask.unsqueeze(2)
                den = mask.sum(dim=1).unsqueeze(1)
                sentence_embeddings.extend(se.sum(dim=1) / den)

            return torch.stack(sentence_embeddings)

        seed_embeddings_bert = [embedder(seed) for seed in list(seeds.values())]

        if load and os.path.isfile(f'{self.root_path}/bert_avg_train_embeddings.pickle'):
            print(f'Loading embeddings from {self.root_path}')
            embeddings_bert = torch.load(f'{self.root_path}/bert_avg_train_embeddings.pickle')
        else:
            embeddings_bert = embedder(self.sentences)
            print(f'Saving embeddings to {self.root_path}')
            torch.save(embeddings_bert, f'{self.root_path}/bert_avg_train_embeddings.pickle')

        return seed_embeddings_bert, embeddings_bert

    def __sbert_embedder(self, load, seeds):
        seed_embeddings = [self.model.encode(seed, convert_to_tensor=True) for seed in list(seeds.values())]

        if load and os.path.isfile(f'{self.root_path}/sbert_train_embeddings.pickle'):
            print(f'Loading embeddings from {self.root_path}')
            embeddings = torch.load(f'{self.root_path}/sbert_train_embeddings.pickle')
        else:
            embeddings = self.model.encode(self.sentences, convert_to_tensor=True, show_progress_bar=True)
            print(f'Saving embeddings to {self.root_path}')
            torch.save(embeddings, f'{self.root_path}/sbert_train_embeddings.pickle')
        return seed_embeddings, embeddings
