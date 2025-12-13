import utils.logging as logging
import torch.optim as optim

logger = logging.get_logger(__name__)

def construct_optimizer(model, cfg):
    """
    Construct an optimizer. 
    Supported optimizers include:
        SGD:    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
        ADAM:   Diederik P.Kingma, and Jimmy Ba. "Adam: A Method for Stochastic Optimization."
        ADAMW:  Ilya Loshchilov, and Frank Hutter. "Decoupled Weight Decay Regularization."
    Args:
        model (model): model for optimization.
        cfg (Config): Config object that includes hyper-parameters for the optimizers. 
    """
    lsnhead_params = []
    lsn_params = []
    generator_params = []
    adapter_params = []
    for pname, p in model.named_parameters():
        if any(k in pname for k in ['Linear', 'up_linear', 'up_ln']):
            p.requires_grad = True
            lsnhead_params +=[p]
        elif any(k in pname for k in ['side']):
            p.requires_grad = True
            lsn_params += [p]
        elif any(k in pname for k in ['generator']):
            p.requires_grad = True
            generator_params += [p]
        elif any(k in pname for k in ['Adapter']):
            p.requires_grad = True
            adapter_params += [p]
        else:
            p.requires_grad = False
    if len(lsnhead_params) == 0: 
        params_group = [
            {'params': lsn_params, 'lr': cfg.SOLVER.LSN_LR}
           ]
    else:
        params_group = [
            {'params': lsnhead_params, 'lr': cfg.SOLVER.LSNHEAD_LR  },
            {'params': generator_params, 'lr': cfg.SOLVER.GENERATOR_LR },
            {'params': lsn_params, 'lr': cfg.SOLVER.LSN_LR },
            {'params': adapter_params, 'lr': cfg.SOLVER.LSN_LR*cfg.SOLVER.ADAPTER_ALPHA }  ]
        print("fc params is {}".format(len(lsnhead_params)))
    if cfg.SOLVER.OPTIM_METHOD == 'adamw':
        optimizer = optim.AdamW(params_group, eps=1e-6, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIM_METHOD == 'sgd':
        print("using a sgd optimzer !")
        optimizer = optim.SGD(params_group, momentum=0.9, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIM_METHOD == 'adam':
        print("using a adam optimzer !")
        optimizer = optim.Adam(params_group)
    return optimizer
