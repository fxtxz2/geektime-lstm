# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
import torchtext
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

from LSTM import LSTM
from train import train, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_iter = torchtext.datasets.IMDB(root='./data', split='train')
    # 创建分词器
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    print(tokenizer('here is the an example!'))

    # 构建词汇表
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    print(vocab(tokenizer('here is the an example <pad> <pad>')))

    # 数据处理pipelines
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: 1 if x == 'pos' else 0

    print(text_pipeline('here is the an example'))

    print(label_pipeline('neg'))

    def collate_batch(batch):
        max_length = 256
        pad = text_pipeline('<pad>')
        label_list, text_list, length_list = [], [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = text_pipeline(_text)[:max_length]
            length_list.append(len(processed_text))
            text_list.append((processed_text + pad * max_length)[:max_length])
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.tensor(text_list, dtype=torch.int64)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        return label_list.to(device), text_list.to(device), length_list.to(device)

    train_dataset = to_map_style_dataset(train_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(train_dataset,
                                              [num_train, len(train_dataset) - num_train])
    train_dataloader = DataLoader(split_train_, batch_size=8, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=8, shuffle=False, collate_fn=collate_batch)

    # 实例化模型
    vocab_size = len(vocab)
    embedding_dim = 300
    hidden_dim = 300
    output_dim = 2
    n_layers = 2
    bidirectional = True
    dropout_rate = 0.5

    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate)
    model = model.to(device)

    # 损失函数与优化方法
    lr = 5e-4
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_epochs = 10
    best_valid_loss = float('inf')

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        valid_losses.extend(valid_loss)
        valid_accs.extend(valid_acc)
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_valid_loss = np.mean(valid_loss)
        epoch_valid_acc = np.mean(valid_acc)
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), 'lstm.pt')
        print(f'epoch: {epoch + 1}')
        print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
        print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
