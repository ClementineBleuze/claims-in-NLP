import numpy as np
import torch
import pandas as pd
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any

from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import BatchAllTripletLoss

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn

# check that a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# connect to huggingface
login(token = "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", add_to_git_credential = True)

with open("train_eval_test_datasets_PLM4SSC.pkl", "rb") as f:
    d = pickle.load(f)
    train_dataset = d["train"]
    eval_dataset = d["eval"]
    test_dataset = d["test"]

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

class SSCModel(nn.Module):

    def __init__(self, bert_checkpoint, input_size, hidden_size, output_size, dropout, nb_sentences):
        super(SSCModel, self).__init__()

        # Load BERT model and tokenizer
        self.bert_model = AutoModel.from_pretrained(bert_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
        # add new special tokens [SEC], [NPS], [NNS] to the tokenizer
        special_tokens_dict = {'additional_special_tokens': ['[SEC]','[NPS]','[NNS]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        self.bert_tokenizer = tokenizer
        # resize the token embeddings matrix of the model
        self.bert_model.resize_token_embeddings(len(tokenizer))

        # Define the MLP part of the model
        self.MLP = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, output_size),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

        self.nb_sentences = nb_sentences

    def forward(self, x):

        ## BERT part
        # The input is plain text that needs to be tokenized
        tokenized_input = self.bert_tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Get the output from the BERT model
        output = self.bert_model(**tokenized_input)
        # Get the last hidden state to get the output vectors
        output_vectors = output.last_hidden_state
        # Get the position of the [SEP] tokens (1 per sentence to be classified)
        # [SEP] token is 103 in BERT's vocabulary
        # print(tokenized_input.input_ids.shape)
        # print(tokenized_input.input_ids)
        sep_ids = []
        for bi in range(tokenized_input.input_ids.shape[0]):
            ids = [i for i in range(len(tokenized_input.input_ids[bi])) if tokenized_input.input_ids[bi][i] == 103]
            # in case we miss some separators (ie if the 3 sentences have more than 512 tokens, we truncate on the right and replace the last tokens with [SEP]
            missing_SEP = self.nb_sentences - len(ids)
            if missing_SEP > 0:
               	tokenized_input_mod = torch.clone(tokenized_input.input_ids)
            for i in range(missing_SEP):
                tokenized_input_mod[bi][512 - 1 - i] = 103
                ids.append(512 - 1 - i)
                
            sep_ids.append(ids)
        #print(sep_ids)

        # Get the output vector for the [SEP] tokens
        sep_vectors = output_vectors[0][sep_ids, :]

        ## MLP part
        # Get the output from the MLP
        output = self.MLP(sep_vectors)

        return output


input_size = 768 # size of BERT embeddings
hidden_size = 100
output_size = 8 # number of classes
nb_sentences = 3 # number of sentences per input
dropout = 0.1
batch_size = 8

model = SSCModel(bert_checkpoint = "allenai/scibert_scivocab_uncased",
                 input_size = input_size,
                 hidden_size = hidden_size,
                 output_size = output_size,
                 nb_sentences = nb_sentences,
                 dropout = dropout,
                )

# initialization (ensure reproducibility: everybody should have the same results)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

torch.manual_seed(0)
model.apply(init_weights)


example_batch = next(iter(train_dataloader))[0]
model(example_batch) # see if the model works

# how many parameters in the model ?
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

print(f'The model has {count_parameters(model):,} trainable parameters')

def train(model, train_dataloader, eval_dataloader, num_epochs, num_steps, loss_fn, learning_rate, eval_strategy = "steps", verbose=True, weight_decay = 0.1):

    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)
    # Set the model in training mode
    model_tr.train()
    
    # # optional: freeze the BERT model
    # for param in model_tr.bert_model.parameters():
    #     param.requires_grad = False

    # print the number of parameters that will be fine-tuned
    print(f'The model has {count_parameters(model_tr):,} trainable parameters')
    print(f'The model has {count_parameters(model_tr.bert_model):,} trainable parameters in the BERT part'
            f' and {count_parameters(model_tr.MLP):,} in the MLP part')
    # print the number of parameters that require grad
    print(f'The model has {sum(p.numel() for p in model_tr.parameters() if p.requires_grad):,} parameters that require grad')
    

    # Define the optimizer
    optimizer = torch.optim.Adam(model_tr.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    # Initialize a list for storing the training and validation loss over epochs / steps (depending on the evaluation strategy)
    train_losses = []
    eval_losses = []

    nb_steps = 0
    total_num_steps = int(len(train_dataloader) * num_epochs / batch_size)

    if eval_strategy == "steps":
        # Initialize the training loss for the current step
        tr_loss = 0
    
    # Training loop
    for epoch in range(num_epochs):
        
        if eval_strategy == "epochs":
            # Initialize the training loss for the current epoch
            tr_loss = 0
        
        # Iterate over batches using the dataloader
        for batch_index, (sequence, labels) in enumerate(tqdm(train_dataloader)):
            
            # TO DO: write the training procedure for each batch. This should consist of:
            # - vectorizing the images (size should be (batch_size, input_size))
            # - calculate the predicted labels from the vectorized images using 'model_tr'
            # - using loss_fn, calculate the 'loss' between the predicted and true labels
            # - set the optimizer gradients at 0 for safety
            # - compute the gradients (use the 'backward' method on 'loss')
            # - apply the gradient descent algorithm (perform a step of the optimizer)

            predicted_labels = model_tr(sequence) # shape [nb_sentences, output_size]
            gold_labels = labels.squeeze(0) # shape [nb_sentences, output_size]

            loss = loss_fn(predicted_labels[:,1,:], gold_labels[:,1,:]) # focus only on the center sentence
            
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            tr_loss += loss.item()
            optimizer.step()

            nb_steps += 1

            if eval_strategy == "steps" and nb_steps % num_steps == 0:
                tr_loss = tr_loss/num_steps # average loss over the steps
                train_losses.append(tr_loss)
                
                ## VALIDATION LOSS
                model_tr.eval()
                eval_loss = 0
                with torch.no_grad():
                    for batch_index, (sequence, labels) in enumerate(eval_dataloader):
                        predicted_labels = model_tr(sequence)
                        gold_labels = labels.squeeze(0)
                        loss = loss_fn(predicted_labels[:,1,:], gold_labels[:,1,:])
                        eval_loss += loss.item()

                eval_loss = eval_loss/len(eval_dataloader.dataset)
                eval_losses.append(eval_loss)
                
                if verbose:
                    print('Step [{}/{}], Training loss: {:.4f}, Validation loss: {:.4f}'.format(nb_steps, total_num_steps, tr_loss, eval_loss))

                # save the model if the validation loss is the best so far
                #if eval_loss == min(eval_losses):
                torch.save(model_tr.state_dict(), f"best_model_{nb_steps}.pth")
                print("Model saved (steps = ", nb_steps, ")")
                
                tr_loss = 0 # reset the training loss
                eval_loss = 0 # reset the validation loss
            
            model_tr.train() # set the model back in training mode


        if eval_strategy == "epochs":
            # At the end of each epoch, get the average training loss and store it
            tr_loss = tr_loss/len(train_dataloader.dataset)
            train_losses.append(tr_loss)

            # Display the training loss
            if verbose:
                print('Epoch [{}/{}], Training loss: {:.4f}'.format(epoch+1, num_epochs, tr_loss))
    
    return model_tr, train_losses, eval_losses

import copy
# use a sample of eval dataloader

model, train_losses, eval_losses = train(
    model = model,
    train_dataloader= train_dataloader,
    eval_dataloader = eval_dataloader,
    num_epochs = 3,
    num_steps = 500,
    eval_strategy = "steps",
    loss_fn = nn.CrossEntropyLoss(),
    learning_rate = 1e-5,
)

plt.plot(range(0, 500*len(train_losses), 500), train_losses, "bo-")
plt.plot(range(0, 500*len(eval_losses), 500), eval_losses, "ro-")
plt.xlabel("Epochs")
plt.ylabel("Loss (Cross Entropy)")
plt.title("Evolution of the training and evaluation loss over steps")
plt.show()



