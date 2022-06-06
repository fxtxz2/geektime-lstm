import torch
import torchtext

from LSTM import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = [vocab[t] for t in tokens]
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    predicted_class_title = ['neg', 'pos']
    return predicted_class_title[predicted_class], predicted_probability

if __name__ == "__main__":
    text = "This film is terrible!"

    train_iter = torchtext.datasets.IMDB(root='./data', split='train')
    # 创建分词器
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')


    # 构建词汇表
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)


    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # 加载模型
    vocab_size = len(vocab)
    embedding_dim = 300
    hidden_dim = 300
    output_dim = 2
    n_layers = 2
    bidirectional = True
    dropout_rate = 0.5
    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate)
    model.load_state_dict(torch.load('./lstm.pt'))
    model.to(device)
    model.eval()
    print(predict_sentiment(text, model, tokenizer, vocab, device))
