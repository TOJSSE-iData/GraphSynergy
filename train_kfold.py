import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.GraphSynergy import GraphSynergy as module_arch
from parse_config import ConfigParser
from trainer.trainer import RegressTrainer

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def calc_stat(numbers):
    mu = sum(numbers) / len(numbers)
    sigma = (sum([(x - mu) ** 2 for x in numbers]) / len(numbers)) ** 0.5
    return mu, sigma


def conf_inv(mu, sigma, n):
    delta = 2.776 * sigma / (n ** 0.5)  # 95%
    return mu - delta, mu + delta


def main(config):
    """Training."""
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_dataset(valid=True)
    test_data_loader = data_loader.split_dataset(test=True)
    feature_index = data_loader.get_feature_index()
    cell_neighbor_set = data_loader.get_cell_neighbor_set()
    drug_neighbor_set = data_loader.get_drug_neighbor_set()
    node_num_dict = data_loader.get_node_num_dict()

    model = module_arch(protein_num=node_num_dict['protein'],
                        cell_num=node_num_dict['cell'],
                        drug_num=node_num_dict['drug'],
                        emb_dim=config['arch']['args']['emb_dim'],
                        n_hop=config['arch']['args']['n_hop'],
                        l1_decay=config['arch']['args']['l1_decay'],
                        therapy_method=config['arch']['args']['therapy_method'])
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = RegressTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      feature_index=feature_index,
                      cell_neighbor_set=cell_neighbor_set,
                      drug_neighbor_set=drug_neighbor_set,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      len_epoch=config['trainer']['epochs'])
    trainer.train()

    """Testing."""
    logger = config.get_logger('test')
    logger.info(model)
    test_metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    # load best checkpoint
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    test_output = trainer.test()
    log = {'loss': test_output['total_loss'] / test_output['n_samples']}
    log.update(test_output['total_metrics'])
    square_errors = [(y-yy)**2 for y, yy in zip(test_output['y_true'], test_output['y_pred'])]
    mu, sigma = calc_stat(square_errors)
    lo, hi = conf_inv(mu, sigma, len(square_errors))
    log.update({'Confidence interval': f"[{lo:.4f}, {hi:.4f}]"})
    logger.info(log)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config/Oneil_config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--fd', '--test_fold'], type=int, target='data_loader;args;test_fold')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
