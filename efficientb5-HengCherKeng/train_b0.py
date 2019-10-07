import os
import numpy
import torch
import cv2
from timeit import default_timer as timer
from .dataset import *
from .model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# TODO: brush up augmentation parameters
# TODO: WTF is infor?
def train_augment(image, label, mask, infor):
    random_augment_flag = numpy.random.choice(3)  # TODO: Why is this random?
    if random_augment_flag == 0:
        pass
    elif random_augment_flag == 1:
        image, mask = do_random_crop_rescale(image, mask, 1600 - (256 - 180), 180)
    elif random_augment_flag == 2:
        image, mask = do_random_crop_rotate_rescale(image, mask, 1600 - (256 - 224), 224)
    image, mask = do_random_crop(image, mask, 400, 256)
    if numpy.random.rand() > 0.25:
        image, mask = do_random_cutout(image, mask)
    if numpy.random.rand() > 0.5:
        image, mask = do_flip_lr(image, mask)
    if numpy.random.rand() > 0.5:
        image, mask = do_flip_ud(image, mask)
    if numpy.random.rand() > 0.5:
        image = do_random_log_contast(image, gain=[0.70, 1.50])
    u = numpy.random.choice(3)
    if u == 0:
        pass
    if u == 1:
        image = do_random_noise(image, noise=8)
    if u == 2:
        image = do_random_salt_pepper_noise(image, noise=0.0001)
    # if   u==3:
    #     image = do_random_salt_pepper_line(image)
    return image, label, mask, infor


def do_valid(net, valid_loader, out_dir=None, debug_flag=None):
    valid_loss = numpy.zeros(17, numpy.float32)
    valid_num = numpy.zeros_like(valid_loss)
    for t, (input, truth_label, truth_mask, truth_attention, infor) in enumerate(valid_loader):
        # if b==5: break
        batch_size = len(infor)
        net.eval()
        input = input.cuda()
        truth_label = truth_label.cuda()
        truth_mask = truth_mask.cuda()
        truth_attention = truth_attention.cuda()
        with torch.no_grad():
            logit_mask = torch.nn.parallel.data_parallel(net, input)
            loss = criterion_mask(logit_mask, truth_mask)
            probability_label = logit_mask_to_probability_label(logit_mask)
            tn, tp, num_neg, num_pos = metric_label(probability_label, truth_label)
            dn, dp, num_neg, num_pos = metric_mask(logit_mask, truth_mask)
        l = numpy.array([loss.item() * batch_size, *tn, *tp, *dn, *dp])
        n = numpy.array([batch_size, *num_neg, *num_pos, *num_neg, *num_pos])
        valid_loss += l
        valid_num += n
        # Dum results for debug
        if debug_flag is None:
            probability_mask = torch.nn.functional.softmax(logit_mask, 1)
            probability_label = probability_label.data.cpu().numpy()
            probability_mask = probability_mask.data.cpu().numpy()
            truth_label = truth_label.data.cpu().numpy()
            truth_mask = truth_mask.data.cpu().numpy()
            image = input_to_image(input)
            for b in range(batch_size):
                image_id = infor[b].image_id
                # TODO: Figure out how to implement draw_predict_result, or find the implementation
                result = draw_predict_result(
                    image[b], truth_label[b], truth_mask[b], probability_label[b], probability_mask[b])
                cv2.image_show('result', result, resize=0.5)
                cv2.imwrite(out_dir + '/valid/%s.png' % image_id[:-4], result)
                cv2.waitKey(1)
                pass
        # print(valid_loss)
        print('\r %4d/%4d' % (valid_num[0], len(valid_loader.dataset)), end='', flush=True)
        pass  # -- end of one data loader --
    assert (valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss / valid_num
    return valid_loss


def run_train(out_dir='/out', initial_checkpoint_fn='/out/checkpoint.pth'):
    sampler = FiveBalanceClassSampler  # RandomSampler #FiveBalanceClassSampler
    loss_weight = None  # [5,10,2,5]
    # TODO: Figure out how to implement NullScheduler, or find the implementation
    scheduler = NullScheduler(lr=0.001)
    iter_accum = 1
    batch_size = 10  # 8
    # Setup
    for f in ['checkpoint', 'train', 'valid', 'backup']:
        os.makedirs(out_dir + '/' + f, exist_ok=True)
    # TODO: Figure out how to implement backup_project_as_zip, Logger,
    # TODO: caps-locked parameters, or find the implementation
    backup_project_as_zip(PROJECT_PATH, out_dir + '/backup/code.train.%s.zip' % IDENTIFIER)
    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    # Dataset
    log.write('** dataset setting **\n')
    train_dataset = SteelDataset(
        mode='train',
        csv=['train.csv', ],
        split=['train_b0_11568.npy', ],
        augment=train_augment,
    )
    train_loader = torch. DataLoader(
        train_dataset,
        sampler=sampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=null_collate
    )
    valid_dataset = SteelDataset(
        mode='train',
        csv=['train.csv', ],
        split=['valid_b0_1000.npy', ],
        augment=None,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=torch.utils.data.sampler.SequentialSampler(valid_dataset),
        batch_size=4,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=null_collate
    )
    assert (len(train_dataset) >= batch_size)
    log.write('batch_size = %d\n' % batch_size)
    log.write('train_dataset : \n%s\n' % train_dataset)
    log.write('valid_dataset : \n%s\n' % valid_dataset)
    log.write('\n')
    # Net
    log.write('** net setting **\n')
    net = Net().cuda()
    log.write('\tinitial_checkpoint_fn = %s\n' % initial_checkpoint_fn)
    if initial_checkpoint_fn is not None:
        state_dict = torch.load(initial_checkpoint_fn, map_location=lambda storage, loc: storage)
        # for k in list(state_dict.keys()):
        #     if any(s in k for s in ['g_block1',]): state_dict.pop(k, None)
        # net.load_state_dict(state_dict,strict=False)
        net.load_state_dict(state_dict, strict=False)  # True
    else:
        net.load_pretrain(is_print=False)
    log.write('%s\n' % (type(net)))
    log.write('loss_weight=%s\n' % (str(loss_weight)))
    log.write('sampler=%s\n' % (str(sampler)))
    log.write('\n')
    # Optimiser
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass
    # net.set_mode('train',is_freeze_bn=True)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=scheduler(0))
    # optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=scheduler(0), momentum=0.9,
                                weight_decay=0.0001)
    num_iters = 3000 * 1000
    iter_smooth = 50
    iter_log = 200
    iter_valid = 200
    iter_save = [0, num_iters - 1] + list(range(0, num_iters, 1000))  # 1*1000
    start_iter = 0
    start_epoch = 0
    rate = 0
    if initial_checkpoint_fn is not None:
        initial_optimizer = initial_checkpoint_fn.replace('_model.pth', '_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint = torch.load(initial_optimizer)
            start_iter = checkpoint['iter']
            start_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
        pass
    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('scheduler\n  %s\n' % (scheduler))
    log.write('\n')
    # Training starts here!
    log.write('** Training starts here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n' % (batch_size, iter_accum))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    log.write(
        '                     |------------------------------------------- VALID------------------------------------------------|---------------------- TRAIN/BATCH ---------------------\n')
    log.write(
        'rate     iter  epoch |  loss           [tn1,2,3,4  :  tp1,2,3,4]                    [dn1,2,3,4  :  dp1,2,3,4]          |  loss    [tn :  tp1,2,3,4]          | time             \n')
    log.write(
        '--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    # 0.00000 135.0*  65.1 |   nan  [1.00 1.00 1.00 1.00 : 0.00 0.00 0.00 0.00]  [1.00 1.00 1.00 1.00 : 0.00 0.00 0.00 0.00] | 0.000  [0.00 : 0.00 0.00 0.00 0.00] |  0 hr 00 min
    valid_loss = numpy.zeros(17, numpy.float32)
    train_loss = numpy.zeros(6, numpy.float32)
    batch_loss = numpy.zeros_like(valid_loss)
    iter = 0
    i = 0
    start = timer()
    while iter < num_iters:
        sum_train_loss = numpy.zeros_like(train_loss)
        sum_train = numpy.zeros_like(train_loss)
        optimizer.zero_grad()
        for t, (input, truth_label, truth_mask, truth_attention, infor) in enumerate(train_loader):
            batch_size = len(infor)
            iter = i + start_iter
            epoch = (iter - start_iter) * batch_size / len(train_dataset) + start_epoch
            # if 0:
            if (iter % iter_valid == 0):
                valid_loss = do_valid(net, valid_loader, out_dir)  #
                pass
            if (iter % iter_log == 0):
                print('\r', end='', flush=True)
                asterisk = '*' if iter in iter_save else ' '
                log.write(
                    '%0.5f %5.1f%s %5.1f | %5.3f  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f]  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f] | %5.3f  [%0.2f : %0.2f %0.2f %0.2f %0.2f] | %s' % ( \
                        rate, iter / 1000, asterisk, epoch,
                        *valid_loss,
                        *train_loss,
                        time_to_str((timer() - start), 'min'))
                )
                log.write('\n')
            # if 0:
            if iter in iter_save:
                torch.save({
                    # 'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'epoch': epoch,
                }, out_dir + '/checkpoint/%08d_optimizer.pth' % (iter))
                if iter != start_iter:
                    torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (iter))
                    pass
            # learning rate scheduler -------------
            lr = scheduler(iter)
            if lr < 0:
                break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)
            # one iteration update  -------------
            # net.set_mode('train',is_freeze_bn=True)
            net.train()
            input = input.cuda()
            truth_label = truth_label.cuda()
            truth_mask = truth_mask.cuda()
            truth_attention = truth_attention.cuda()
            logit_mask = data_parallel(net, input)
            loss = criterion_mask(logit_mask, truth_mask)
            probability_label = logit_mask_to_probability_label(logit_mask)
            tn, tp, num_neg, num_pos = metric_label(probability_label, truth_label)
            (loss / iter_accum).backward()
            if (iter % iter_accum) == 0:
                optimizer.step()
                optimizer.zero_grad()
            # print statistics  --------
            l = numpy.array([loss.item() * batch_size, tn.sum(), *tp])
            n = numpy.array([batch_size, num_neg.sum(), *num_pos])
            batch_loss = l / n
            sum_train_loss += l
            sum_train += n
            if iter % iter_smooth == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train[...] = 0
            print('\r', end='', flush=True)
            asterisk = ' '
            print(
                '%0.5f %5.1f%s %5.1f | %5.3f  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f]  [%0.2f %0.2f %0.2f %0.2f : %0.2f %0.2f %0.2f %0.2f] | %5.3f  [%0.2f : %0.2f %0.2f %0.2f %0.2f] | %s' % ( \
                    rate, iter / 1000, asterisk, epoch,
                    *valid_loss,
                    *batch_loss,
                    time_to_str((timer() - start), 'min'))
                , end='', flush=True)
            i = i + 1
            # Debug
            if 0:
                for di in range(3):
                    if (iter + di) % 1000 == 0:

                        probability_attention = torch.sigmoid(logit_attention)
                        probability_attention = probability_attention.data.cpu().numpy().reshape(-1, 4, 5)
                        truth_label = truth_label.data.cpu().numpy()
                        truth_mask = truth_mask.data.cpu().numpy()
                        truth_attention = truth_attention.data.cpu().numpy().reshape(-1, 4, 5)

                        image = input_to_image(input)
                        for b in range(batch_size):
                            image_id = infor[b].image_id
                            result = draw_predict_result_8cls(image[b], truth_label[b], truth_mask[b],
                                                              truth_attention[b], probability_attention[b])

                            image_show('result', result, resize=0.5)
                            cv2.imwrite(out_dir + '/train/%05d.png' % (di * 100 + b), result)
                            cv2.waitKey(1)
                            pass
        pass  # -- end of one data loader --
    pass  # -- end of all iterations --
    log.write('\n')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_train()
