"""
lumen/train.py
Author: Tingfeng Xia

Pytorch Training Entry Point. 
"""


from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import loaders
import utils
import random
from options import Opt

parser = argparse.ArgumentParser(
    prog='lumen trainer',
    description='training for lumen: emotion classification with eye area only'
)

parser.add_argument(
    '--model', type=str, default="VGGLIKE_SM"
)

args = parser.parse_args()


# for reproducability. set on first import
random.seed(405728300)
np.random.seed(405728300)
torch.manual_seed(405728300)

# keep: best results
best_test_acc = 0  # best test accuracy
best_test_acc_epoch = 0

# initialize options (load from arguments)
opt = Opt(model=args.model)
print(opt)

# initialize the train and test loaders
trainloader = loaders.initialize_train_loader_with_opt(opt)
testloader = loaders.initialize_test_loader_with_opt(opt)

# initialize the model
net = loaders.initialize_model_with_opt(opt)
if opt.usecuda:
    net.cuda()

# initialize loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(),
    lr=opt.lr,
    momentum=.9,
    weight_decay=5e-4
)


def train(epoch):
    print('\n---\nEpoch: %d' % (epoch + 1))
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > opt.lr_decay_start and opt.lr_decay_start >= 0:
        frac = (epoch - opt.lr_decay_start) // opt.lr_decay_every
        decay_factor = opt.lr_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if opt.usecuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_acc = 100.*correct/total
    return train_acc, train_loss


def test(epoch):
    global test_acc
    global best_test_acc
    global best_test_acc_epoch

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if opt.usecuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            outputs_avg = outputs.view(
                bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            test_loss += loss.data
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    test_acc = 100.*correct/total

    if test_acc > best_test_acc:
        print('Saving..')
        print("best_test_acc: %0.3f" % max(test_acc, best_test_acc))
        state = {
            'net': net.state_dict() if opt.usecuda else net,
            'best_test_acc': test_acc,
            'best_test_acc_epoch': epoch,
        }
        if not os.path.isdir(opt.savepath):
            os.mkdir(opt.savepath)
        torch.save(state, os.path.join(opt.savepath, 'best_model.t7'))

        best_test_acc = test_acc
        best_test_acc_epoch = epoch

    return test_acc, test_loss


if __name__ == "__main__":
    # training loop
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(opt.total_epoch):
        train_acc, train_loss = train(epoch)
        test_acc, test_loss = test(epoch)

    print("best_test_acc: %0.3f" % best_test_acc)
    print("best_test_acc_epoch: %d" % best_test_acc_epoch)

    # save train and test losses
    np.save(os.path.join(opt.savepath, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(opt.savepath, 'test_losses.npy'), np.array(test_losses))
    np.save(os.path.join(opt.savepath, 'train_accs.npy'), np.array(train_accs))
    np.save(os.path.join(opt.savepath, 'test_accs.npy'), np.array(test_accs))
