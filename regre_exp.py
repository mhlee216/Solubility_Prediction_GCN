import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'Epoch': epoch,
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)

def train(model, device, optimizer, train_loader, criterion, args):
    epoch_train_loss = 0
    for i, [solute, solvent] in enumerate(train_loader):
        solute = solute.to(device)
        solvent = solvent.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(solute, solvent, device)
        output.require_grad = False
        train_loss = criterion(output, solute.y.view(-1,1))
        epoch_train_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
    epoch_train_loss /= len(train_loader)
    print('- Loss : %.4f' % epoch_train_loss)
    return model, epoch_train_loss

def test(model, device, test_loader, args):
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        logS_total = list()
        pred_logS_total = list()
        for i, [solute, solvent] in enumerate(test_loader):
            solute = solute.to(device)
            solvent = solvent.to(device)
            logS_total += solute.y.tolist()
            output = model(solute, solvent, device)
            pred_logS_total += output.view(-1).tolist()
            y_pred_list.append(output.cpu().numpy())
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        mae = mean_absolute_error(logS_total, pred_logS_total)
        std = np.std(np.array(logS_total)-np.array(pred_logS_total))
        mse = mean_squared_error(logS_total, pred_logS_total)
        r_square = r2_score(logS_total, pred_logS_total)
    print()
    print('[Test]')
    print('- MAE : %.4f' % mae)
    print('- MSE : %.4f' % mse)
    print('- R2 : %.4f' % r_square)
    return mae, std, mse, r_square, logS_total, pred_logS_total, y_pred_list

def experiment(model, train_loader, test_loader, device, args):
    time_start = time.time()
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)
    
    list_train_loss = list()
    print('[Train]')
    for epoch in range(args.epoch):
        scheduler.step()
        print('- Epoch :', epoch+1)
        model, train_loss = train(model, device, optimizer, train_loader, criterion, args)
        list_train_loss.append(train_loss)
    
    mae, std, mse, r_square, logS_total, pred_logS_total, y_pred_list = test(model, device, test_loader, args)
    
    time_end = time.time()
    time_required = time_end - time_start
    
    args.list_train_loss = list_train_loss
    args.logS_total = logS_total
    args.pred_logS_total = pred_logS_total
    args.mae = mae
    args.std = std
    args.mse = mse
    args.r_square = r_square
    args.time_required = time_required
    args.y_pred_list = y_pred_list
    
    save_checkpoint(epoch, model, optimizer, args.model_path)
    
    return args