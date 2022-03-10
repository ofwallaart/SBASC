from transformers import AutoTokenizer
from models.SBASC.model import BERTLinear
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import time
import math


class Trainer:
    def __init__(self, cfg, learning_rate, beta1, beta2, batch_size, gamma1, gamma2):
        self.domain = cfg.domain.name
        self.bert_type = cfg.domain.bert_mapper
        self.device = cfg.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.root_path = cfg.path_mapper
        self.batch_size = batch_size
        self.epochs = cfg.epochs
        self.validation_data_size = cfg.domain.validation_data_size
        self.hyper_validation_size = cfg.domain.hyper_validation_size

        categories = cfg.domain.aspect_category_mapper
        polarities = cfg.domain.sentiment_category_mapper

        self.model = BERTLinear(self.bert_type, len(
            categories), len(polarities), gamma1, gamma2).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta1, beta2))

        aspect_dict = {}
        inv_aspect_dict = {}
        for i, cat in enumerate(categories):
            aspect_dict[i] = cat
            inv_aspect_dict[cat] = i

        polarity_dict = {}
        inv_polarity_dict = {}
        for i, pol in enumerate(polarities):
            polarity_dict[i] = pol
            inv_polarity_dict[pol] = i

        self.aspect_dict = aspect_dict
        self.inv_aspect_dict = inv_aspect_dict
        self.polarity_dict = polarity_dict
        self.inv_polarity_dict = inv_polarity_dict

    def load_training_data(self):
        sentences = []
        cats = []
        pols = []
        with open(f'{self.root_path}/label-sentences.txt', 'r', encoding="utf8") as f:
            for idx, line in enumerate(f):
                if idx % 2 == 1:
                    cat, pol = line.strip().split()
                    cats.append(self.inv_aspect_dict[cat])
                    pols.append(self.inv_polarity_dict[pol])
                else:
                    sentences.append(line.strip())
        encoded_dict = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt',
            max_length=128,
            return_attention_mask=True,
            truncation=True)
        labels_cat = torch.tensor(cats)
        labels_pol = torch.tensor(pols)
        dataset = TensorDataset(
            labels_cat, labels_pol, encoded_dict['input_ids'], encoded_dict['token_type_ids'],
            encoded_dict['attention_mask'])
        return dataset

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def train_model(self, dataset, epochs=None, hyper=False):
        """Train the model.
            """
        self.set_seed(0)

        if epochs is None:
            epochs = self.epochs

        # Prepare dataset
        if hyper:
            data_size = math.floor(len(dataset) * self.hyper_validation_size)
            train_data, val_data = torch.utils.data.random_split(
                dataset, [data_size, len(dataset) - data_size])
        else:
            train_data, val_data = torch.utils.data.random_split(
                dataset, [len(dataset) - self.validation_data_size, self.validation_data_size])
        dataloader = DataLoader(train_data, batch_size=self.batch_size)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)

        model = self.model
        device = self.device

        optimizer = self.optimizer

        best_loss = 0

        # Start training loop
        print("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(
                f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()

            for step, (labels_cat, labels_pol, input_ids, token_type_ids, attention_mask) in enumerate(dataloader):
                batch_counts += 1

                optimizer.zero_grad()
                encoded_dict = {
                    'input_ids': input_ids.to(device),
                    'token_type_ids': token_type_ids.to(device),
                    'attention_mask': attention_mask.to(device)
                }
                loss, _, _ = model(labels_cat.to(device),
                                   labels_pol.to(device), **encoded_dict)

                # Perform a backward pass to calculate gradients
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                total_loss += loss.item()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 50 == 0 and step != 0) or (step == len(dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            avg_train_loss = total_loss / len(dataloader)

            print("-" * 70)

            model.eval()

            # Tracking variables
            val_accuracy = []
            val_loss = []

            with torch.no_grad():

                for labels_cat, labels_pol, input_ids, token_type_ids, attention_mask in val_dataloader:
                    encoded_dict = {
                        'input_ids': input_ids.to(device),
                        'token_type_ids': token_type_ids.to(device),
                        'attention_mask': attention_mask.to(device)
                    }
                    loss, logits_cat, logits_pol = model(labels_cat.to(
                        device), labels_pol.to(device), **encoded_dict)
                    val_loss.append(loss.item())

                    # Get the predictions
                    preds_cat = torch.argmax(logits_cat, dim=1).flatten()
                    preds_col = torch.argmax(logits_pol, dim=1).flatten()

                    # Calculate the accuracy rate
                    accuracy = (preds_cat == labels_cat.to(device)).cpu().numpy().mean() * 100
                    val_accuracy.append(accuracy)

            # Compute the average accuracy and loss over the validation set.
            val_loss = np.mean(val_loss)
            val_accuracy = np.mean(val_accuracy)

            if val_loss > best_loss:
                torch.save(self.model, f'{self.root_path}/model.pth')

            # Display the epoch training loss and validation loss
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")
        return val_loss, val_accuracy

    def save_model(self, name):
        model = torch.load(f'{self.root_path}/model.pth')
        torch.save(model, f'{self.root_path}/{name}.pth')

    def load_model(self, name):
        self.model = torch.load(f'{self.root_path}/{name}.pth')

    def evaluate(self):
        test_sentences = []
        test_cats = []
        test_pols = []

        with open(f'{self.root_path}/test.txt', 'r', encoding="utf8") as f:
            for line in f:
                _, cat, pol, sentence = line.strip().split('\t')
                cat = int(cat)
                pol = int(pol)
                test_cats.append(cat)
                test_pols.append(pol)
                test_sentences.append(sentence)

        df = pd.DataFrame(columns=(
            ['sentence', 'actual category', 'predicted category', 'actual polarity', 'predicted polarity']))

        model = torch.load(f'{self.root_path}/model.pth')
        model.eval()
        device = self.device

        actual_aspect = []
        predicted_aspect = []

        actual_polarity = []
        predicted_polarity = []

        iters = 0
        with torch.no_grad():
            for input, cat, pol in tqdm(zip(test_sentences, test_cats, test_pols)):
                encoded_dict = self.tokenizer([input],
                                              padding=True,
                                              return_tensors='pt',
                                              return_attention_mask=True,
                                              truncation=True).to(device)

                loss, logits_cat, logits_pol = model(torch.tensor([cat]).to(
                    device), torch.tensor([pol]).to(device), **encoded_dict)

                actual_aspect.append(self.aspect_dict[cat])
                actual_polarity.append(self.polarity_dict[pol])

                predicted_aspect.append(
                    self.aspect_dict[torch.argmax(logits_cat).item()])
                predicted_polarity.append(
                    self.polarity_dict[torch.argmax(logits_pol).item()])
                df.loc[iters] = [input, actual_aspect[-1], predicted_aspect[-1],
                                 actual_polarity[-1], predicted_polarity[-1]]
                iters += 1

        df.to_csv(f'{self.root_path}/predictions.csv')

        predicted_pol = np.array(predicted_polarity)
        actual_pol = np.array(actual_polarity)
        print("Polarity")
        print(classification_report(actual_pol, predicted_pol, digits=4))
        print()

        predicted = np.array(predicted_aspect)
        actual = np.array(actual_aspect)
        print("Aspect")
        print(classification_report(actual, predicted, digits=4))
        print()

        return classification_report(actual_pol, predicted_pol, digits=6, output_dict=True), classification_report(
            actual, predicted, digits=6, output_dict=True), predicted_pol, predicted
