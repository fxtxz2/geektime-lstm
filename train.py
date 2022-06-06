import sys

import torch
import tqdm
from visdom import Visdom


def train(dataloader, model, criterion, optimizer, device):
    # 实例化一个窗口
    viz = Visdom(port=8097)
    # 初始化窗口的信息
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

    model.train()
    epoch_losses = []
    epoch_accs = []
    for i, batch in enumerate(tqdm.tqdm(dataloader, desc='training...', file=sys.stdout)):
        (label, ids, length) = batch
        label = label.to(device)
        ids = ids.to(device)
        length = length.to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label) # loss计算
        accuracy = get_accuracy(prediction, label)
        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
        # 更新监听的信息
        viz.line([loss.item()], [i], win='train_loss', update='append')
    return epoch_losses, epoch_accs

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            (label, ids, length) = batch
            label = label.to(device)
            ids = ids.to(device)
            length = length.to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label) # loss计算
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return epoch_losses, epoch_accs

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy