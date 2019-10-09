import os
import logging
import torch
import datetime
import numpy
import timeit
from . import model
from .dataset import samplers, dataset, augmentations


def do_valid(net, valid_loader, out_dir=None, debug_flag=None):
    valid_loss = numpy.zeros(17, numpy.float32)
    valid_num = numpy.zeros_like(valid_loss)
    for t, (input, truth_label, truth_mask, truth_attention, infor) in enumerate(
        valid_loader
    ):
        # if b==5: break
        batch_size = len(infor)
        net.eval()
        input = input.cuda()
        truth_label = truth_label.cuda()
        truth_mask = truth_mask.cuda()
        truth_attention = truth_attention.cuda()
        with torch.no_grad():
            logit_mask = torch.nn.parallel.data_parallel(net, input)
            loss = model.criterion_mask(logit_mask, truth_mask)
            probability_label = model.logit_mask_to_probability_label(logit_mask)
            tn, tp, num_neg, num_pos = model.metric_label(probability_label, truth_label)
            dn, dp, num_neg, num_pos = model.metric_mask(logit_mask, truth_mask)
        l = numpy.array([loss.item() * batch_size, *tn, *tp, *dn, *dp])
        n = numpy.array([batch_size, *num_neg, *num_pos, *num_neg, *num_pos])
        valid_loss += l
        valid_num += n
        print(
            "\r %4d/%4d" % (valid_num[0], len(valid_loader.dataset)), end="", flush=True
        )
        pass
    assert valid_num[0] == len(valid_loader.dataset)
    valid_loss = valid_loss / valid_num
    return valid_loss


def train_b0_network(out_dir="/out", initial_checkpoint_fn="/out/checkpoint.pth", torch_seed=0, iter_accum=1, batch_size=10, init_lr=0.001):
    # Fix PyTorch seed
    torch.manual_seed(torch_seed)
    # Setup
    for f in ["checkpoint", "train", "valid", "backup"]:
        os.makedirs(out_dir + "/" + f, exist_ok=True)
    log = logging.Logger()
    log.open(out_dir + "/log.train.txt", mode="a")
    log.write("\n--- [START %s] %s\n\n" % (datetime.datetime.now(), "-" * 64))
    log.write("\tPyTorch seed = %u\n" % torch_seed)
    log.write("\t__file__     = %s\n" % __file__)
    log.write("\tout_dir      = %s\n" % out_dir)
    log.write("\n")
    # Dataset
    log.write("** dataset setting **\n")
    train_dataset = dataset.SteelDataset(
        mode="train",
        csv=["train.csv"],
        split=["train_b0_11568.npy"],
        augment=augmentations.train_augment,
    )
    # We have several samplers to use, located in dataset/samplers.py
    sampler = samplers.FiveBalanceClassSampler(train_dataset)
    train_loader = torch.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=augmentations.null_collate,
    )
    valid_dataset = dataset.SteelDataset(
        mode="train",
        csv=["train.csv"],
        split=["valid_b0_1000.npy"],
        augment=None
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=torch.utils.data.sampler.SequentialSampler(valid_dataset),
        batch_size=4,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=augmentations.null_collate,
    )
    assert len(train_dataset) >= batch_size
    log.write("batch_size = %d\n" % batch_size)
    log.write("train_dataset : \n%s\n" % train_dataset)
    log.write("valid_dataset : \n%s\n" % valid_dataset)
    log.write("\n")
    # Net
    log.write("** net setting **\n")
    net = model.Net().cuda()
    log.write("\tinitial_checkpoint_fn = %s\n" % initial_checkpoint_fn)
    if initial_checkpoint_fn is not None:
        state_dict = torch.load(
            initial_checkpoint_fn, map_location=lambda storage, loc: storage
        )
        net.load_state_dict(state_dict, strict=False)
    else:
        net.load_pretrain(is_print=False)
    log.write("%s\n" % (type(net)))
    log.write("sampler=%s\n" % (str(sampler)))
    log.write("\n")
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=init_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer
    )
    num_iters = 3000 * 1000
    iter_smooth = 50
    iter_log = 200
    iter_valid = 200
    iter_save = [0, num_iters - 1] + list(range(0, num_iters, 1000))
    start_iter = 0
    start_epoch = 0
    rate = 0
    if initial_checkpoint_fn is not None:
        initial_optimizer = initial_checkpoint_fn.replace(
            "_model.pth", "_optimizer.pth"
        )
        if os.path.exists(initial_optimizer):
            checkpoint = torch.load(initial_optimizer)
            start_iter = checkpoint["iter"]
            start_epoch = checkpoint["epoch"]
        pass
    log.write("optimizer\n  %s\n" % optimizer)
    log.write("scheduler\n  %s\n" % scheduler)
    log.write("\n")
    # Training starts here!
    log.write("** Training starts here! **\n")
    log.write("   batch_size=%d,  iter_accum=%d\n" % (batch_size, iter_accum))
    log.write("   experiment  = %s\n" % str(__file__.split("/")[-2:]))
    log.write(
        "                     |------------------------------------------- VALID------------------------------------------------|---------------------- TRAIN/BATCH ---------------------\n"
    )
    log.write(
        "rate     iter  epoch |  loss           [tn1,2,3,4  :  tp1,2,3,4]                    [dn1,2,3,4  :  dp1,2,3,4]          |  loss    [tn :  tp1,2,3,4]          | time             \n"
    )
    log.write(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
    )
    valid_loss = numpy.zeros(17, numpy.float32)
    train_loss = numpy.zeros(6, numpy.float32)
    batch_loss = numpy.zeros_like(valid_loss)
    iter = 0
    i = 0
    start = timeit.default_timer()
    while iter < num_iters:
        sum_train_loss = numpy.zeros_like(train_loss)
        sum_train = numpy.zeros_like(train_loss)
        optimizer.zero_grad()
        for t, (input, truth_label, truth_mask, truth_attention, infor) in enumerate(
            train_loader
        ):
            batch_size = len(infor)
            iter = i + start_iter
            epoch = (iter - start_iter) * batch_size / len(train_dataset) + start_epoch
            # if 0:
            if iter % iter_valid == 0:
                valid_loss = do_valid(net, valid_loader, out_dir)  #
                pass
            if iter % iter_log == 0:
                print("\r", end="", flush=True)
                asterisk = "*" if iter in iter_save else " "
                log.write(
                    "%0.5f %5.1f%s %5.1f | %5.3f  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f]  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f] | %5.3f  [%0.2f : %0.2f %0.2f %0.2f %0.2f] | %s"
                    % (
                        rate,
                        iter / 1000,
                        asterisk,
                        epoch,
                        *valid_loss,
                        *train_loss,
                        # TODO: Replace this with cleaner code
                        str((timeit.default_timer() - start) // 60) + str(timeit.default_timer() - 60 * (timeit.default_timer() // 60)),
                    )
                )
                log.write("\n")
            if iter in iter_save:
                torch.save(
                    {
                        "iter": iter,
                        "epoch": epoch,
                    },
                    out_dir + "/checkpoint/%08d_optimizer.pth" % iter,
                )
                if iter != start_iter:
                    torch.save(
                        net.state_dict(),
                        out_dir + "/checkpoint/%08d_model.pth" % iter,
                    )
                    pass
            # Learning rate scheduler
            lr = scheduler(iter)
            if lr < 0:
                break
            net.train()
            input = input.cuda()
            truth_label = truth_label.cuda()
            truth_mask = truth_mask.cuda()
            truth_attention = truth_attention.cuda()
            logit_mask = torch.nn.parallel.data_parallel(net, input)
            loss = model.criterion_mask(logit_mask, truth_mask)
            probability_label = model.logit_mask_to_probability_label(logit_mask)
            tn, tp, num_neg, num_pos = model.metric_label(probability_label, truth_label)
            (loss / iter_accum).backward()
            if (iter % iter_accum) == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            # Print statistics
            l = numpy.array([loss.item() * batch_size, tn.sum(), *tp])
            n = numpy.array([batch_size, num_neg.sum(), *num_pos])
            batch_loss = l / n
            sum_train_loss += l
            sum_train += n
            if iter % iter_smooth == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train[...] = 0
            print("\r", end="", flush=True)
            asterisk = " "
            print(
                "%0.5f %5.1f%s %5.1f | %5.3f  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f]  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f] | %5.3f  [%0.2f : %0.2f %0.2f %0.2f %0.2f] | %s"
                % (
                    rate,
                    iter / 1000,
                    asterisk,
                    epoch,
                    *valid_loss,
                    *batch_loss,
                    str((timeit.default_timer() - start) // 60) + str(timeit.default_timer() - 60 * (timeit.default_timer() // 60)),
                ),
                end="",
                flush=True,
            )
            i = i + 1
        pass
    pass
    log.write("\n")


if __name__ == "__main__":
    print("%s: calling main function ... " % os.path.basename(__file__))
    train_b0_network()
