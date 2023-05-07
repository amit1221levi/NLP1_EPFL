from torch.utils.data import Dataset
import datasets
import torch
import torch.nn as nn

# define the RNN dataset class
class RNNDataset(Dataset):
    def __init__(self, dataset, max_seq_length):
        self.texts = dataset
        self.max_seq_length = max_seq_length
        # define the dataset
        dataset_length = len(dataset)
        train_length = int(dataset_length * 0.8)
        test_length = dataset_length - train_length

        # split the dataset into train and test sets
        rnn_train_dataset, rnn_test_dataset = torch.utils.data.random_split(dataset['text'],
                                                               [train_length, test_length],
                                                               generator=torch.Generator().manual_seed(0))

        # define the vocabulary
        vocab = set()
        for sentence in rnn_train_dataset:
            for word in sentence.split():
                vocab.add(word)
        vocab_size = len(vocab)

        # create a dictionary that maps each word to a unique index
        word_to_index = {word: index for index, word in enumerate(vocab)}
        index_to_word = {index: word for word, index in word_to_index.items()}

    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # split the sentence into words and convert each word to its index
        indexed_text = [self.word_to_index[word] for word in text.split() if word in self.vocab]
        # pad the sequence if its length is less than max_seq_length
        if len(indexed_text) < self.max_seq_length:
            indexed_text += [0] * (self.max_seq_length - len(indexed_text))
        # truncate the sequence if its length is greater than max_seq_length
        indexed_text = indexed_text[:self.max_seq_length]
        # convert the list to a tensor
        indexed_text = torch.tensor(indexed_text)
        # return the indexed sentence and its label
        return indexed_text, indexed_text[1:]

    


class VanillaLSTM(nn.Module):
    def __init__(self, vocab_size, 
                 embedding_dim,
                 hidden_dim,
                 num_layers, 
                 dropout_rate,
                 embedding_weights=None,
                 freeze_embeddings=False):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # pass embeeding weights if exist
        if embedding_weights is not None:
            # YOUR CODE HERE
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=freeze_embeddings)

            pass

        else:  # train from scratch embeddings
            # YOUR CODE HERE
            self.embedding = nn.Embedding(vocab_size, embedding_dim)


        # YOUR CODE HERE
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_id):
       # YOUR CODE HERE
        x = self.embedding(input_id)
        output, (hidden, cell) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size):
        super(EncoderDecoder, self).__init__()
        # YOUR CODE HERE
        self.encoder = nn.LSTM(input_vocab_size, hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_vocab_size)


    def forward(self, inputs, input_mask, targets=None):
        # YOUR CODE HERE
        encoder_output, (hidden_state, cell_state) = self.encoder(inputs)
        outputs = self.decoder(hidden_state[-1])
        return outputs
