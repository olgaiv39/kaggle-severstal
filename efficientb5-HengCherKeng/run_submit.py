import os
import cv2
import numpy
import torch
from timeit import default_timer as timer
from .dataset import *
from .model import *
from .etc import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
TEMPERATE = 0.5


def probability_mask_to_probability_label(probability):
    batch_size, num_class, H, W = probability.shape
    probability = probability.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 5)
    value, index = probability.max(1)
    probability = value[:, 1:]
    return probability


def remove_small_one(predict, min_size):
    H, W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(numpy.uint8))
    predict = numpy.zeros((H, W), numpy.bool)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = True
    return predict


def remove_small(predict, min_size):
    for b in range(len(predict)):
        for c in range(4):
            predict[b, c] = remove_small_one(predict[b, c], min_size[c])
    return predict


def do_evaluate_segmentation(net, test_dataset, augment=[], out_dir=None):
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=torch.utils.data.sampler.SequentialSampler(test_dataset),
        batch_size=2,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate
    )

    # TODO: WTF is this function defined at this scope?
    def sharpen(p, t=TEMPERATE):
        if t != 0:
            return p ** t
        else:
            return p

    test_num = 0
    test_id = []
    # test_image = []
    test_probability_label = []  # 8bit
    test_probability_mask = []  # 8bit
    test_truth_label = []
    test_truth_mask = []
    start = timer()
    for t, (input, truth_label, truth_mask, truth_attention, infor) in enumerate(test_loader):
        batch_size, C, H, W = input.shape
        input = input.cuda()
        with torch.no_grad():
            net.eval()
            num_augment = 0
            if 1:  # null
                logit = torch.nn.parallel.data_parallel(net, input)  # net(input)
                probability = torch.softmax(logit, 1)
                probability_mask = sharpen(probability, 0)
                num_augment += 1
            if 'flip_lr' in augment:
                logit = torch.nn.parallel.data_parallel(net, torch.flip(input, dims=[3]))
                probability = torch.softmax(torch.flip(logit, dims=[3]), 1)
                probability_mask += sharpen(probability)
                num_augment += 1
            if 'flip_ud' in augment:
                logit = torch.nn.parallel.data_parallel(net, torch.flip(input, dims=[2]))
                probability = torch.softmax(torch.flip(logit, dims=[2]), 1)
                probability_mask += sharpen(probability)
                num_augment += 1
            probability_mask = probability_mask / num_augment
            probability_label = probability_mask_to_probability_label(probability_mask)
        batch_size = len(infor)
        truth_label = truth_label.data.cpu().numpy().astype(numpy.uint8)
        truth_mask = truth_mask.data.cpu().numpy().astype(numpy.uint8)
        probability_mask = (probability_mask.data.cpu().numpy() * 255).astype(numpy.uint8)
        probability_label = (probability_label.data.cpu().numpy() * 255).astype(numpy.uint8)
        test_id.extend([i.image_id for i in infor])
        test_probability_mask.append(probability_mask)
        test_probability_label.append(probability_label)
        test_truth_mask.append(truth_mask)
        test_truth_label.append(truth_label)
        test_num += batch_size
        print('\r %4d / %4d  %s' % (
            test_num, len(test_loader.dataset), time_to_str((timer() - start), 'min')
        ), end='', flush=True)
    assert (test_num == len(test_loader.dataset))
    print('')
    start_timer = timer()
    test_probability_mask = numpy.concatenate(test_probability_mask)
    test_probability_label = numpy.concatenate(test_probability_label)
    test_truth_mask = numpy.concatenate(test_truth_mask)
    test_truth_label = numpy.concatenate(test_truth_label)
    # TODO: implement time_to_str
    # print(time_to_str((timer() - start_timer), 'sec'))
    return test_id, test_truth_label, test_truth_mask, test_probability_label, test_probability_mask


def run_submit_segmentation():
    out_dir = \
        '/root/share/project/kaggle/2019/steel/result15/efficientb5-fpn-crop256x400-foldb1'
    initial_checkpoint = \
        '/root/share/project/kaggle/2019/steel/result15/efficientb5-fpn-crop256x400-foldb1/checkpoint/00053000_model.pth'
    train_split = ['valid_b1_1000.npy', ]
    out_dir = \
        '/root/share/project/kaggle/2019/steel/result15/efficientb5-fpn-crop256x400-foldb0h'
    initial_checkpoint = \
        '/root/share/project/kaggle/2019/steel/result15/efficientb5-fpn-crop256x400-foldb0h/checkpoint/00051000_model.pth'
    train_split = ['valid_b0_1000.npy', ]
    mode = 'test'  # 'train' # 'test'
    augment = ['null']  # ['null', 'flip_lr','flip_ud']  #
    # Setup
    os.makedirs(out_dir + '/submit/%s' % (mode), exist_ok=True)
    # TODO: Find logger and define parameters
    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
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
    if mode == 'train':
        test_dataset = SteelDataset(
            mode='train',
            csv=['train.csv', ],
            split=train_split,
            augment=None,
        )
    if mode == 'test':
        test_dataset = SteelDataset(
            mode='test',
            csv=['sample_submission.csv', ],
            split=['test_1801.npy', ],
            augment=None,  #
        )
    log.write('test_dataset : \n%s\n' % (test_dataset))
    log.write('\n')
    # Net
    log.write('** net setting **\n')
    net = Net().cuda()
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage), strict=True)
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n' % (type(net)))
    log.write('\n')
    # Testing starts here!
    if 0:
        # Save
        image_id, truth_label, truth_mask, probability_label, probability_mask, = \
            do_evaluate_segmentation(net, test_dataset, augment)
        if 1:
            # Save
            write_list_to_file(out_dir + '/submit/%s/image_id.txt' % (mode), image_id)
            numpy.savez_compressed(out_dir + '/submit/%s/truth_label.uint8.npz' % (mode), truth_label)
            numpy.savez_compressed(out_dir + '/submit/%s/truth_mask.uint8.npz' % (mode), truth_mask)
            numpy.savez_compressed(out_dir + '/submit/%s/probability_label.uint8.npz' % (mode), probability_label)
            numpy.savez_compressed(out_dir + '/submit/%s/probability_mask.uint8.npz' % (mode), probability_mask)
    if 1:
        image_id = read_list_from_file(out_dir + '/submit/%s/image_id.txt' % (mode))
        truth_label = numpy.load(out_dir + '/submit/%s/truth_label.uint8.npz' % (mode))['arr_0']
        truth_mask = numpy.load(out_dir + '/submit/%s/truth_mask.uint8.npz' % (mode))['arr_0']
        probability_label = numpy.load(out_dir + '/submit/%s/probability_label.uint8.npz' % (mode))['arr_0']
        probability_mask = numpy.load(out_dir + '/submit/%s/probability_mask.uint8.npz' % (mode))['arr_0']
    if 1:
        # Decode
        num_test = len(image_id)
        value = numpy.max(probability_mask, 1, keepdims=True)
        value = probability_mask * (value == probability_mask)
        # Remove background class
        probability_mask = probability_mask[:, 1:]
        index = numpy.ones((num_test, 4, 256, 1600), numpy.uint8) * numpy.array([1, 2, 3, 4], numpy.uint8).reshape(1, 4, 1, 1)
        truth_mask = truth_mask == index
    threshold_label = [0.75, 0.75, 0.50, 0.50, ]
    threshold_mask_pixel = [0.40, 0.40, 0.40, 0.40, ]
    threshold_mask_size = [1, 1, 1, 1, ]
    # Inspect here!
    print('')
    log.write('submitting .... @ %s\n' % str(augment))
    log.write('threshold_label = %s\n' % str(threshold_label))
    log.write('threshold_mask_pixel = %s\n' % str(threshold_mask_pixel))
    log.write('threshold_mask_size  = %s\n' % str(threshold_mask_size))
    log.write('\n')
    if mode == 'train':
        predict_label = probability_label > (numpy.array(threshold_label) * 255).astype(numpy.uint8).reshape(1, 4)
        predict_mask = probability_mask > (numpy.array(threshold_mask_pixel) * 255).astype(numpy.uint8).reshape(1, 4, 1, 1)
        log.write('** threshold_label **\n')
        kaggle, result = compute_metric_label(truth_label, predict_label)
        text = summarise_metric_label(kaggle, result)
        log.write('\n%s' % (text))
        auc, result = compute_roc_label(truth_label, probability_label)
        text = summarise_roc_label(auc, result)
        log.write('\n%s' % (text))
        log.write('** threshold_pixel **\n')
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s' % (text))
        log.write('** threshold_pixel + threshold_label **\n')
        predict_mask = predict_mask * predict_label.reshape(-1, 4, 1, 1)
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s' % (text))
        log.write('** threshold_pixel + threshold_label + threshold_size **\n')
        predict_mask = remove_small(predict_mask, threshold_mask_size)
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s' % (text))
    if mode == 'test':
        log.write('test submission .... @ %s\n' % str(augment))
        csv_file = out_dir + '/submit/%s/efficientnetb5-fpn.csv' % (mode)
        predict_label = probability_label > (numpy.array(threshold_label) * 255).astype(numpy.uint8).reshape(1, 4)
        predict_mask = probability_mask > (numpy.array(threshold_mask_pixel) * 255).astype(numpy.uint8).reshape(1, 4, 1, 1)
        image_id_class_id = []
        encoded_pixel = []
        for b in range(len(image_id)):
            for c in range(4):
                image_id_class_id.append(image_id[b] + '_%d' % (c + 1))
                if predict_label[b, c] == 0:
                    rle = ''
                else:
                    rle = run_length_encode(predict_mask[b, c])
                encoded_pixel.append(rle)
        df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
        df.to_csv(csv_file, index=False)
        # Print statistics
        text = summarise_submission_csv(df)
        log.write('\n')
        log.write('%s' % (text))
    exit(0)


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_submit_segmentation()
