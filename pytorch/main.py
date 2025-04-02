from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40_Merged
from model import DGCNN_PN_Merge
from util import cal_loss, IOStream
import numpy as np
import sklearn.metrics as metrics
from torch.utils.data import DataLoader

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(f'checkpoints/{args.exp_name}'):
        os.makedirs(f'checkpoints/{args.exp_name}')
    if not os.path.exists(f'checkpoints/{args.exp_name}/models'):
        os.makedirs(f'checkpoints/{args.exp_name}/models')
    os.system(f'cp main.py checkpoints/{args.exp_name}/main.py.backup')
    os.system(f'cp model.py checkpoints/{args.exp_name}/model.py.backup')
    os.system(f'cp util.py checkpoints/{args.exp_name}/util.py.backup')
    os.system(f'cp data.py checkpoints/{args.exp_name}/data.py.backup')

def train(args, io):
    train_loader = DataLoader(ModelNet40_Merged(partition='train', num_points=args.num_points),
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40_Merged(partition='test', num_points=args.num_points),
                             num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN_PN_Merge(args).to(device)
    io.cprint(str(model))

    model = nn.DataParallel(model)
    io.cprint(f"Using {torch.cuda.device_count()} GPUs")

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    criterion = cal_loss

    best_test_acc = 0

    for epoch in range(args.epochs):
        scheduler.step()

        model.train()
        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []

        for csh, osr, label in train_loader:
            csh, osr, label = csh.to(device), osr.to(device), label.to(device).squeeze()
            csh = csh.permute(0, 2, 1)
            osr = osr.permute(0, 2, 1)

            opt.zero_grad()
            logits = model(csh, osr)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()

            preds = logits.max(dim=1)[1]
            count += csh.size(0)
            train_loss += loss.item() * csh.size(0)
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        io.cprint(f"Train {epoch}, loss: {train_loss/count:.6f}, acc: {metrics.accuracy_score(train_true, train_pred):.6f}, avg acc: {metrics.balanced_accuracy_score(train_true, train_pred):.6f}")

        # Evaluation
        model.eval()
        test_loss = 0.0
        count = 0.0
        test_pred = []
        test_true = []

        for csh, osr, label in test_loader:
            csh, osr, label = csh.to(device), osr.to(device), label.to(device).squeeze()
            csh = csh.permute(0, 2, 1)
            osr = osr.permute(0, 2, 1)

            logits = model(csh, osr)
            loss = criterion(logits, label)

            preds = logits.max(dim=1)[1]
            count += csh.size(0)
            test_loss += loss.item() * csh.size(0)
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        io.cprint(f"Test {epoch}, loss: {test_loss/count:.6f}, acc: {test_acc:.6f}, avg acc: {avg_per_class_acc:.6f}")

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'checkpoints/{args.exp_name}/models/model.t7')

def test(args, io):
    test_loader = DataLoader(ModelNet40_Merged(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN_PN_Merge(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_pred = []
    test_true = []

    for csh, osr, label in test_loader:
        csh, osr, label = csh.to(device), osr.to(device), label.to(device).squeeze()
        csh = csh.permute(0, 2, 1)
        osr = osr.permute(0, 2, 1)
        logits = model(csh, osr)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    io.cprint(f'Test :: acc: {test_acc:.6f}, avg acc: {avg_per_class_acc:.6f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition with DGCNN + PointNet Merge')
    parser.add_argument('--exp_name', type=str, default='exp_merge', help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--use_sgd', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--model_path', type=str, default='')

    args = parser.parse_args()
    _init_()

    io = IOStream(f'checkpoints/{args.exp_name}/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(f'Using GPU : {torch.cuda.current_device()} from {torch.cuda.device_count()} devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)


