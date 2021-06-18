import torch
import torch.utils.data as data
import torchnet as tnt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import os
import json
import pickle as pkl
import argparse
import pprint
from tqdm import tqdm
from models.stclassifier import PseTae, PseLTae, PseGru, PseTempCNN
from dataset import PixelSetData, PixelSetData_preloaded
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis

import re
import collections.abc
from torch.nn import functional as F

def train_epoch(model, optimizer, criterion, data_loader, device, config):
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for i, (x, y, dates) in enumerate(data_loader):
        y_true.extend(list(map(int, y)))

        x = recursive_todevice(x, device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x, batch_positions=dates)
        loss = criterion(out, y.long())
        loss.backward()
        optimizer.step()

        pred = out.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}'.format(i + 1, len(data_loader), loss_meter.value()[0],
                                                                    acc_meter.value()[0]))

    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_accuracy': acc_meter.value()[0],
                     'train_IoU': mIou(y_true, y_pred, n_classes=config['num_classes'])}

    return epoch_metrics


def evaluation(model, criterion, loader, device, config, mode='val'):
    y_true = []
    y_pred = []

    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    for (x, y, dates) in loader:
    # for (x, y) in tqdm(loader):
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x, dates)
            loss = criterion(prediction, y)

        acc_meter.add(prediction, y)
        loss_meter.add(loss.item())

        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))

    metrics = {'{}_accuracy'.format(mode): acc_meter.value()[0],
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_IoU'.format(mode): mIou(y_true, y_pred, config['num_classes'])}

    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes'])))

np_str_obj_array_pattern = re.compile(r'[SaUO]')

def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)

def pad_collate(batch):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError('Format not managed : {}'.format(elem.dtype))
            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError('Format not managed : {}'.format(elem_type))

def get_test_loaders(dt, config):
    test_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                  num_workers=config['num_workers'], drop_last=True)
    return test_loader

def get_loaders(dt, kfold, config):
    indices = list(range(len(dt)))
    np.random.shuffle(indices)

    kf = KFold(n_splits=kfold, shuffle=False)
    indices_seq = list(kf.split(list(range(len(dt)))))
    ntest = len(indices_seq[0][1])

    loader_seq = []
    for trainval, test_indices in indices_seq:
        trainval = [indices[i] for i in trainval]
        test_indices = [indices[i] for i in test_indices]

        validation_indices = trainval[-ntest:]
        train_indices = trainval[:-ntest]

        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        validation_sampler = data.sampler.SubsetRandomSampler(validation_indices)
        test_sampler = data.sampler.SubsetRandomSampler(test_indices)

        train_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                       sampler=train_sampler,
                                       num_workers=config['num_workers'], drop_last=True)
        validation_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                            sampler=validation_sampler,
                                            num_workers=config['num_workers'], drop_last=True)
        test_loader = data.DataLoader(dt, batch_size=config['batch_size'],
                                      sampler=test_sampler,
                                      num_workers=config['num_workers'], drop_last=True)

        loader_seq.append((train_loader, validation_loader, test_loader))
    return loader_seq


def get_multi_years_loaders(dt, kfold, config):
    indices = [list(range(i*len(dt[i]), (i + 1)*len(dt[i]))) for i in range(len(dt))]
    for i in range(len(indices)):
        np.random.shuffle(indices[i])

    kf = KFold(n_splits=kfold, shuffle=False)
    indices_seq = [list(kf.split(list(range(len(i))))) for i in dt]
    ntest = len(indices_seq[0][0][1])

    loader_seq = []
    test_loader = [[] for i in range(kfold)]
    validation_indices = [[] for i in range(kfold)]
    train_indices = [[] for i in range(kfold)]
    merged_dt = dt[0]
    for i in range(1, len(dt)):
        # merged_dt.dates += dt[i].dates
        # merged_dt.date_positions += dt[i].date_positions
        merged_dt.date_positions.update(dt[i].date_positions)
        merged_dt.pid += dt[i].pid
        merged_dt.target += dt[i].target

    for idx, dataset in enumerate(dt):
        for id_fold, (trainval, test_indices) in enumerate(indices_seq[idx]):
            trainval = [indices[idx][i] for i in trainval]
            test_indices = [indices[idx][i] for i in test_indices]

            test_sampler = data.sampler.SubsetRandomSampler(test_indices)

            test_loader[id_fold].append(data.DataLoader(merged_dt, batch_size=config['batch_size'],
                                          sampler=test_sampler,
                                          num_workers=config['num_workers'], drop_last=True, collate_fn=pad_collate))
            validation_indices[id_fold] += trainval[-ntest:]
            train_indices[id_fold] += trainval[:-ntest]
    for i in range(kfold):
        train_sampler = data.sampler.SubsetRandomSampler(train_indices[i])
        validation_sampler = data.sampler.SubsetRandomSampler(validation_indices[i])

        train_loader = data.DataLoader(merged_dt, batch_size=config['batch_size'],
                                       sampler=train_sampler,
                                       num_workers=config['num_workers'], drop_last=True, collate_fn=pad_collate)
        validation_loader = data.DataLoader(merged_dt, batch_size=config['batch_size'],
                                            sampler=validation_sampler,
                                            num_workers=config['num_workers'], drop_last=True, collate_fn=pad_collate)

        loader_seq.append((train_loader, validation_loader, test_loader[i]))
    return loader_seq, merged_dt

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)
    for fold in range(1, config['kfold'] + 1):
        os.makedirs(os.path.join(config['res_dir'], 'Fold_{}'.format(fold)), exist_ok=True)


def checkpoint(fold, log, config):
    with open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, conf_mat, config):
    with open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'conf_mat.pkl'), 'wb'))


def overall_performance(config):
    cm = np.zeros((config['num_classes'], config['num_classes']))
    for fold in range(1, config['kfold'] + 1):
        cm += pkl.load(open(os.path.join(config['res_dir'], 'Fold_{}'.format(fold), 'conf_mat.pkl'), 'rb'))

    _, perf = confusion_matrix_analysis(cm)

    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))

    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as file:
        file.write(json.dumps(perf, indent=4))


def main(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    prepare_output(config)

    mean_std = pkl.load(open(config['dataset_folder'] + '/normvals_tot.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None

    # We only consider the subset of classes with more than 100 samples in the S2-Agri dataset
    # subset = [1, 3, 4, 5, 6, 8, 9, 12, 13, 14, 16, 18, 19, 23, 28, 31, 33, 34, 36, 39]
    # subset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    subset = None
    if config['preload']:
        dt = PixelSetData_preloaded(config['dataset_folder'], labels='CODE9_2018', npixel=config['npixel'],
                                    sub_classes=subset,
                                    norm=mean_std,
                                    extra_feature=extra)
    else:
        dt = []
        for year in config['year']:
            dt.append(PixelSetData(config['dataset_folder'], labels='CODE9_' + year, npixel=config['npixel'],
                              sub_classes=subset,
                              norm=mean_std,
                              extra_feature=extra, year=year))
    device = torch.device(config['device'])

    if len(config['year']) > 0:

        loaders, dt = get_multi_years_loaders(dt, config['kfold'], config)
        for fold, (train_loader, val_loader, test_loader) in enumerate(loaders):
            print('Starting Fold {}'.format(fold + 1))
            print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader[0])))
            model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                               mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                               dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                               positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                               mlp4=config['mlp4'], d_model=config['d_model'])
            if config['geomfeat']:
                model_config.update(with_extra=True, extra_size=4)
            else:
                model_config.update(with_extra=False, extra_size=None)
            model = PseLTae(**model_config)
            config['N_params'] = model.param_ratio()
            with open(os.path.join(config['res_dir'], 'conf.json'), 'w') as file:
                file.write(json.dumps(config, indent=4))

            model = model.to(device)
            model.apply(weight_init)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = FocalLoss(config['gamma'])

            trainlog = {}

            best_mIoU = 0
            for epoch in range(1, config['epochs'] + 1):
                print('EPOCH {}/{}'.format(epoch, config['epochs']))

                model.train()
                train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)

                print('Validation . . . ')
                model.eval()
                val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

                print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(val_metrics['val_loss'], val_metrics['val_accuracy'],
                                                                     val_metrics['val_IoU']))

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(fold + 1, trainlog, config)

                if val_metrics['val_IoU'] >= best_mIoU:
                    best_mIoU = val_metrics['val_IoU']
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))

            print('Testing best epoch . . .')

            for year, i in enumerate(test_loader):
                print("Année: {} ".format(config['year'][year]))
                model_config['len_max_seq'] = config['sly'][config['year'][year]]
                model_test = PseLTae(**model_config)
                config['N_params'] = model_test.param_ratio()
                with open(os.path.join('conf.json'), 'w') as file:
                    file.write(json.dumps(config, indent=4))
                model_test = model_test.to(device)
                model_test.apply(weight_init)
                criterion = FocalLoss(config['gamma'])
                new_state_dict = torch.load(os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))
                model_dict = {k: v for k, v in model_test.state_dict().items() if
                              k != 'temporal_encoder.position_enc.weight'}
                model_dict_copy = {k: v for k, v in model.state_dict().items()}
                compatible_dict = {k: v for k, v in new_state_dict['state_dict'].items() if k in model_dict}
                model_dict_copy.update(compatible_dict)

                model_test.load_state_dict(model_dict_copy)

                model_test.eval()

                test_metrics, conf_mat = evaluation(model_test, criterion, i, device=device, mode='test', config=config)

                print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                                     test_metrics['test_IoU']))
            save_results(fold + 1, test_metrics, conf_mat, config)
            # 1 fold (no cross validation)
            # break


    elif config['test_mode']:
        loader = get_test_loaders(dt, config)
        model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                            mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                            dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                            positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                            mlp4=config['mlp4'], d_model=config['d_model'])
        if config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)
        else:
            model_config.update(with_extra=False, extra_size=None)
        model = PseLTae(**model_config)
        config['N_params'] = model.param_ratio()
        with open(os.path.join( 'conf.json'), 'w') as file:
            file.write(json.dumps(config, indent=4))
        model = model.to(device)
        model.apply(weight_init)
        criterion = FocalLoss(config['gamma'])

        new_state_dict = torch.load(config['loaded_model'])
        model_dict = {k:v for k,v in model.state_dict().items() if k!='temporal_encoder.position_enc.weight'}
        model_dict_copy = {k: v for k, v in model.state_dict().items()}
        compatible_dict = {k: v for k, v in new_state_dict['state_dict'].items() if k in model_dict}
        model_dict_copy.update(compatible_dict)

        model.load_state_dict(model_dict_copy)
        model.eval()

        test_metrics, conf_mat = evaluation(model, criterion, loader, device=device, mode='test', config=config)

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                             test_metrics['test_IoU']))
        save_results(1, test_metrics, conf_mat, config)

    else:
        dt = dt[0]
        loaders = get_loaders(dt, config['kfold'], config)
        for fold, (train_loader, val_loader, test_loader) in enumerate(loaders):
            print('Starting Fold {}'.format(fold + 1))
            print('Train {}, Val {}, Test {}'.format(len(train_loader), len(val_loader), len(test_loader)))

            if config['tae']:
                model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                                    mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                                    dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                                    positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                                    mlp4=config['mlp4'], d_model=config['d_model'])
                if config['geomfeat']:
                    model_config.update(with_extra=True, extra_size=4)
                else:
                    model_config.update(with_extra=False, extra_size=None)
                model = PseTae(**model_config)


            elif config['gru']:
                model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                                    mlp2=config['mlp2'], hidden_dim=config['hidden_dim'],
                                    positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                                    mlp4=config['mlp4'])
                if config['geomfeat']:
                    model_config.update(with_extra=True, extra_size=4)
                else:
                    model_config.update(with_extra=False, extra_size=None)
                model = PseGru(**model_config)

            elif config['tcnn']:
                model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                                    mlp2=config['mlp2'], nker=config['nker'], mlp3=config['mlp3'],
                                    positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                                    mlp4=config['mlp4'])
                if config['geomfeat']:
                    model_config.update(with_extra=True, extra_size=4)
                else:
                    model_config.update(with_extra=False, extra_size=None)
                model = PseTempCNN(**model_config)


            else:
                model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                                    mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                                    dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                                    positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                                    mlp4=config['mlp4'], d_model=config['d_model'])
                if config['geomfeat']:
                    model_config.update(with_extra=True, extra_size=4)
                else:
                    model_config.update(with_extra=False, extra_size=None)
                model = PseLTae(**model_config)

            config['N_params'] = model.param_ratio()
            with open(os.path.join(config['res_dir'], 'conf.json'), 'w') as file:
                file.write(json.dumps(config, indent=4))

            model = model.to(device)
            model.apply(weight_init)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = FocalLoss(config['gamma'])

            trainlog = {}

            best_mIoU = 0
            for epoch in range(1, config['epochs'] + 1):
                print('EPOCH {}/{}'.format(epoch, config['epochs']))

                model.train()
                train_metrics = train_epoch(model, optimizer, criterion, train_loader, device=device, config=config)

                print('Validation . . . ')
                model.eval()
                val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')

                print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(val_metrics['val_loss'], val_metrics['val_accuracy'],
                                                                     val_metrics['val_IoU']))

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(fold + 1, trainlog, config)

                if val_metrics['val_IoU'] >= best_mIoU:
                    best_mIoU = val_metrics['val_IoU']
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                               os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))

            print('Testing best epoch . . .')
            model.load_state_dict(
                torch.load(os.path.join(config['res_dir'], 'Fold_{}'.format(fold + 1), 'model.pth.tar'))['state_dict'])
            model.eval()

            test_metrics, conf_mat = evaluation(model, criterion, test_loader, device=device, mode='test', config=config)

            print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(test_metrics['test_loss'], test_metrics['test_accuracy'],
                                                                 test_metrics['test_IoU']))
            save_results(fold + 1, test_metrics, conf_mat, config)
            # 1 fold (no cross validation)
            break
    overall_performance(config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--dataset_folder', default='', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--year', default=['2018', '2019', '2020'], type=str,
                        help='The year of the data you want to use')
    parser.add_argument('--res_dir', default='./results', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='If specified, the whole dataset is loaded to RAM at initialization')
    parser.set_defaults(preload=False)

    parser.add_argument('--test_mode', default=False, type=bool,
                        help='Load a pre-trained model and test on the whole data set')
    parser.add_argument('--loaded_model', default='/home/FQuinton/Bureau/results/2019_geomfeat_1_20_class_affine/Fold_1/model.pth.tar', type=str,
                        help='Path to the pre-trained model')
    # Training parameters
    parser.add_argument('--kfold', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--npixel', default=64, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[132,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', default=1, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')

    ## L-TAE
    parser.add_argument('--n_head', default=16, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=8, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--mlp3', default='[256,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=36, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--sly', default={'2018': 36, '2019': 27, '2020': 29},
                        help='Sequence length by year for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')
    parser.add_argument('--d_model', default=256, type=int,
                        help="size of the embeddings (E), if input vectors are of a different size, a linear layer is used to project them to a d_model-dimensional space"
                        )

    ## Classifier
    parser.add_argument('--num_classes', default=20, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 20]', type=str, help='Number of neurons in the layers of MLP4')

    ## Other methods (use one of the flags tae/gru/tcnn to train respectively a TAE, GRU or TempCNN instead of an L-TAE)
    ## see paper appendix for hyperparameters
    parser.add_argument('--tae', dest='tae', action='store_true',
                        help="Temporal Attention Encoder for temporal encoding")

    parser.add_argument('--gru', dest='gru', action='store_true', help="Gated Recurent Unit for temporal encoding")
    parser.add_argument('--hidden_dim', default=156, type=int, help="Hidden state size")

    parser.add_argument('--tcnn', dest='tcnn', action='store_true', help="Temporal Convolutions for temporal encoding")
    parser.add_argument('--nker', default='[32,32,128]', type=str, help="Number of successive convolutional kernels ")

    parser.set_defaults(gru=False, tcnn=False, tae=False)

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k or k == 'nker':
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    pprint.pprint(config)
    main(config)
