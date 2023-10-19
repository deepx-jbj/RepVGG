# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------
import time
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from train.config import get_config
from data import build_loader
from train.lr_scheduler import build_scheduler
from train.logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, save_latest, update_model_ema, unwrap_model
import copy
from train.optimizer import build_optimizer
from repvggplus import create_RepVGGplus_by_name

import onnx
from dx_com.onnx.util import proto2session

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
    
    
import torch
import numpy as np



def parse_option():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--model_file', required=True, type=str, help='weight file name')

    # easy config modification
    parser.add_argument('--arch', default=None, type=str, help='arch name')
    parser.add_argument('--batch-size', default=128, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='/mnt/datasets/ILSVRC2012', type=str, help='path to dataset')
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],  #TODO Note: use amp if you have it
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='train_results', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def repvgg_model_convert(model:torch.nn.Module, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    # if save_path is not None:
    #     torch.save(model.state_dict(), save_path)
    return model



def main(config, model_file:str):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    model = create_RepVGGplus_by_name(config.MODEL.ARCH, deploy=False, use_checkpoint=args.use_checkpoint)
    
    # make model same as inference (jbj)
    model.load_state_dict(torch.load(config.MODEL.ARCH + ".pth"))
    model = repvgg_model_convert(model)
    
    optimizer = build_optimizer(config, model)
    
    model.cuda()
    
    if model_file.endswith("pth"):
        from dl_adquantizer import DXQmasterManager
        
        model.eval()
        manager = DXQmasterManager(model, torch.randn(1, 3, 320, 320).cuda(), print_graph=True)
        manager.prepare_model_for_calib()

        # - calibration - #
        model.eval()
        CALNUM = 1
        cnt = 0
        with torch.no_grad():
            for iter, (images, targets) in enumerate(data_loader_val):
                images = images.cuda(non_blocking=True)
                for b in range(len(images)):
                    cnt += 1
                    one_img = images[b:b+1]
                    
                    model(one_img)
                    
                    if cnt % 10 == 0:
                        print(f"clibration cnt: {cnt}")
                    
                    if cnt == CALNUM:
                        break
                if cnt == CALNUM:
                    break
        
        # - calibration end - # -
        manager.prepare_model_for_train(reparam=False)
        manager.model.load_state_dict(torch.load(model_file))
        # compile and export
        save_path = "test.onnx"
        manager.compile(save_path=save_path)
        model = onnx.load(save_path)
    elif model_file.endswith("onnx"):
        onnx.load(model_file)

    if data_loader_val is not None:
        acc1, acc5, loss = validate(config, data_loader_val, model)


@torch.no_grad()
def validate(config, data_loader, model:onnx.ModelProto):
    session = proto2session(model, gpu_mode=True)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        outputs = []
        with torch.no_grad():
            for b in len(images):
                one_img = images[b:b+1].numpy()
                output = torch.tensor(model(images))
                outputs.append(output)
        output = torch.cat(outputs, dim=0)
        
        #   =============================== deepsup part
        if type(output) is dict:
            output = output['main']

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg



import os

if __name__ == '__main__':
    args, config = parse_option()

    main(config, model_file=args.model_file)
