import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

from datasets import get_dataset
from models import DeepLab_ResNet
from utils import CrossEntropyLoss2d


def get_1x_lr_params(model):
    b = []
    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=10, max_iter=30000, power=0.9):
    if iter % lr_decay_iter or iter > max_iter:
        return None

    new_lr = init_lr * (1 - iter / max_iter)**power
    optimizer.params_groups[0]['lr'] = new_lr
    optimizer.params_groups[1]['lr'] = 10 * new_lr


def main(args):
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f)

    # TensorBoard Logger
    writer = SummaryWriter(args.log_dir)

    # Dataset
    dataset = get_dataset(args.dataset)(
        root=config['dataset'][args.dataset]['root'],
        split='train',
        image_size=(321, 321),
        scale=True,
        flip=True,
        preload=True
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=config['num_workers'],
        shuffle=True
    )

    # Model
    model = DeepLab_ResNet(
        n_classes=config['dataset'][args.dataset]['n_classes'])
    state_dict = torch.load(config['dataset'][args.dataset]['init_model'])
    if config['dataset'][args.dataset]['n_classes'] != 21:
        for i in state_dict:
            # 'Scale.layer5.conv2d_list.3.weight'
            i_parts = i.split('.')
            if i_parts[1] == 'layer5':
                state_dict[i] = model.state_dict()[i]
    model.load_state_dict(state_dict)
    if args.cuda:
        model.cuda()

    # Optimizer
    optimizer = {
        'sgd': torch.optim.SGD(
            params=[
                {'params': get_1x_lr_params(model), 'lr': args.lr},
                {'params': get_10x_lr_params(model), 'lr': 10 * args.lr}
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        ),
    }.get(args.optimizer)

    # Loss definition
    criterion = CrossEntropyLoss2d(
        ignore_index=config['dataset'][args.dataset]['ignore_label'])
    if args.cuda:
        criterion.cuda()

    best_loss = 1e10
    patience = args.patience

    # Training loop
    iteration = 0
    for epoch in range(args.n_epoch):
        loss_meter = AverageValueMeter()

        tqdm_loader = tqdm(
            enumerate(loader),
            total=len(loader),
            desc='Epoch [%d]' % (epoch),
            leave=False
        )

        model.train()
        for i, (data, target) in tqdm_loader:
            if args.cuda:
                data = data.cuda()
                target = target.cuda()

            data = Variable(data)
            target = Variable(target)

            # Polynomial lr decay
            poly_lr_scheduler(optimizer=optimizer,
                              init_lr=args.lr,
                              iter=iteration,
                              lr_decay_iter=args.lr_decay,
                              max_iter=len(loader) / args.batch_size * 2,
                              power=args.poly_power)

            # Forward propagation
            optimizer.zero_grad()
            outputs = model(data)
            loss = 0
            for output in outputs:
                output = F.upsample(output, size=321, mode='bilinear')
                loss += criterion(output, target)
            loss_meter.add(loss.data[0], data.size(0))

            # Back propagation & weight updating
            loss.backward()
            optimizer.step()

            # In each 100 iterations
            if iteration % 100 == 0:
                train_loss = loss_meter.value()[0]

                # Early stopping procedure
                if train_loss < best_loss:
                    torch.save(
                        {'iteration': iteration,
                         'weight': model.state_dict()},
                        osp.join(args.save_dir, 'checkpoint_best.pth.tar')
                    )
                    writer.add_text('log', 'Saved a model', iteration)
                    best_loss = train_loss
                    patience = args.patience
                else:
                    patience -= 1
                    if patience == 0:
                        writer.add_text('log', 'Early stopping', iteration)
                        break

                # TensorBoard: Scalar
                writer.add_scalar('train_loss', train_loss, iteration)
                loss_meter.reset()

            iteration += 1


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--dataset', nargs='?', type=str, default='cocostuff')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100)
    parser.add_argument('--batch_size', nargs='?', type=int, default=2)
    parser.add_argument('--lr', nargs='?', type=float, default=0.00025)
    parser.add_argument('--lr_decay', nargs='?', type=int, default=10)
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9)
    parser.add_argument('--weight_decay', nargs='?', type=float, default=5e-4)
    parser.add_argument('--poly_power', nargs='?', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--log_dir', type=str, default='runs')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    for arg in vars(args):
        print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))

    main(args)
