import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax


def train_class_batch(model, samples, targets, criterion):
    # Move samples to the same device as model
    device = next(model.parameters()).device
    samples = samples.to(device)
    targets = targets.to(device)
    
    outputs = model(samples)
    loss = criterion(outputs, targets)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                     max_norm=0, update_freq=1, mixup_fn=None, log_writer=None,
                     args=None, fp16=True, start_steps=None, lr_schedule_values=None,
                     wd_schedule_values=None, num_training_steps_per_epoch=None,
                     update_grad=True, model_ema=None, progress_log_interval=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args is not None and hasattr(args, 'distributed') and args.distributed:
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Starting training loop for epoch {epoch} with {len(data_loader)} batches")

    # Clear CUDA cache at the beginning of each epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared at the beginning of epoch")

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Log progress at specified intervals
        if data_iter_step % progress_log_interval == 0:
            print(f"Processing batch {data_iter_step}/{len(data_loader)} in epoch {epoch}")
            
            # Periodically clear CUDA cache
            if torch.cuda.is_available() and data_iter_step > 0 and data_iter_step % 50 == 0:
                torch.cuda.empty_cache()
                print(f"CUDA cache cleared at step {data_iter_step}")
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
                it = data_iter_step // update_freq + start_steps
                if lr_schedule_values is not None:
                    for i, param_group in enumerate(optimizer.param_groups):
                        if lr_schedule_values is not None:
                            param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and len(wd_schedule_values) > it:
                    for i, param_group in enumerate(optimizer.param_groups):
                        if param_group["weight_decay"] > 0:
                            param_group["weight_decay"] = wd_schedule_values[it]

        try:
            samples = batch[0]
            targets = torch.tensor(batch[1])
            print(f"Loaded batch {data_iter_step} with {len(samples)} samples")

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            if fp16:
                with torch.cuda.amp.autocast():
                    loss, output = train_class_batch(model, samples, targets, criterion)
            else: 
                loss, output = train_class_batch(model, samples, targets, criterion)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= update_freq
            grad_norm = None
            if fp16:
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order,
                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
            else:
                loss.backward()
                if (data_iter_step + 1) % update_freq == 0:
                    if max_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)

            torch.cuda.synchronize()
            
            # Print progress information
            if data_iter_step % progress_log_interval == 0:
                print(f"Completed batch {data_iter_step}/{len(data_loader)}, loss: {loss_value:.4f}")

            metric_logger.update(loss=loss_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            if grad_norm is not None:
                metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                if grad_norm is not None:
                    log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()
                
        except Exception as e:
            print(f"Error processing batch {data_iter_step}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Clear CUDA cache when an error occurs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared after error")
                
            continue

    # Final CUDA cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared at the end of epoch")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
