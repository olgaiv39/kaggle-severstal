# https://github.com/junfu1115/DANet

from common import *
from dataset import *
from efficientnet import *

# overwrite ...
from dataset import null_collate as null_collate0


def null_collate(batch):
    input, truth_label, truth_mask, infor = null_collate0(batch)
    with torch.no_grad():
        arange = torch.FloatTensor([1, 2, 3, 4]).to(truth_mask.device).view(1, 4, 1, 1).long()
        truth_attention = truth_mask.repeat(1, 4, 1, 1)
        truth_attention = (truth_attention == arange).float()
        truth_attention = F.avg_pool2d(truth_attention, kernel_size=(32, 32), stride=(32, 32))
        truth_attention = (truth_attention > 0 / (32 * 32)).float()

    return input, truth_label, truth_mask, truth_attention, infor


####################################################################################################

class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, num_group=32, kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride,
                              bias=False)
        self.gn = nn.GroupNorm(num_group, out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


def upsize_add(x, lateral):
    return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral


def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x


'''
model.py: calling main function ... 


stem   torch.Size([10, 48, 128, 128])
block1 torch.Size([10, 24, 128, 128])

block2 torch.Size([10, 40, 64, 64])

block3 torch.Size([10, 64, 32, 32])

block4 torch.Size([10, 128, 16, 16])
block5 torch.Size([10, 176, 16, 16])

block6 torch.Size([10, 304, 8, 8])
block7 torch.Size([10, 512, 8, 8])
last   torch.Size([10, 2048, 8, 8])

sucess!
'''


class Net(nn.Module):
    def load_pretrain(self, skip=['logit.'], is_print=True):
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        e = EfficientNetB5(drop_connect_rate)
        self.stem = e.stem
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        self.block5 = e.block5
        self.block6 = e.block6
        self.block7 = e.block7
        self.last = e.last
        e = None  # dropped

        # ---
        self.lateral0 = nn.Conv2d(2048, 64, kernel_size=1, padding=0, stride=1)
        self.lateral1 = nn.Conv2d(176, 64, kernel_size=1, padding=0, stride=1)
        self.lateral2 = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        self.lateral3 = nn.Conv2d(40, 64, kernel_size=1, padding=0, stride=1)

        self.top1 = nn.Sequential(
            ConvGnUp2d(64, 64),
            ConvGnUp2d(64, 64),
            ConvGnUp2d(64, 64),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d(64, 64),
            ConvGnUp2d(64, 64),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d(64, 64),
        )
        self.top4 = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.logit_mask = nn.Conv2d(64, num_class + 1, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.stem(x)  # ; print('stem  ',x.shape)
        x = self.block1(x);
        x0 = x  # ; print('block1',x.shape)
        x = self.block2(x);
        x1 = x  # ; print('block2',x.shape)
        x = self.block3(x);
        x2 = x  # ; print('block3',x.shape)
        x = self.block4(x)  # ; print('block4',x.shape)
        x = self.block5(x);
        x3 = x  # ; print('block5',x.shape)
        x = self.block6(x)  # ; print('block6',x.shape)
        x = self.block7(x)  # ; print('block7',x.shape)
        x = self.last(x);
        x4 = x  # ; print('last  ',x.shape)

        # segment
        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3))  # 16x16
        t2 = upsize_add(t1, self.lateral2(x2))  # 32x32
        t3 = upsize_add(t2, self.lateral3(x1))  # 64x64

        t1 = self.top1(t1)  # 128x128
        t2 = self.top2(t2)  # 128x128
        t3 = self.top3(t3)  # 128x128

        t = torch.cat([t1, t2, t3], 1)
        t = self.top4(t)
        logit_mask = self.logit_mask(t)
        logit_mask = F.interpolate(logit_mask, scale_factor=2.0, mode='bilinear', align_corners=False)

        return logit_mask


#########################################################################

# use topk
# def criterion_label(logit, truth, weight=None):
#     batch_size,num_class,H,W = logit.shape
#     K=5
#
#     logit = logit.view(batch_size,num_class,-1)
#     value, index = logit.topk(K)
#
#     logit_k = torch.gather(logit,dim=2,index=index)
#     truth_k = truth.view(batch_size,num_class,1).repeat(1,1,5)
#
#
#     if weight is None: weight=[1,1,1,1]
#     weight = torch.FloatTensor(weight).to(truth.device).view(1,-1,1)
#
#
#     loss = F.binary_cross_entropy_with_logits(logit_k, truth_k, reduction='none')
#     #https://arxiv.org/pdf/1909.07829.pdf
#     if 1:
#         gamma=2.0
#         p = torch.sigmoid(logit_k)
#         focal = (truth_k*(1-p) + (1-truth_k)*(p))**gamma
#         weight = weight*focal /focal.sum().item()
#
#     loss = loss*weight
#     loss = loss.mean()
#     return loss


# use top only
# def criterion_label(logit, truth, weight=None):
#     batch_size,num_class,H,W = logit.shape
#     logit = F.adaptive_max_pool2d(logit,1).view(-1,4)
#     truth = truth.view(-1,4)
#
#     if weight is None: weight=[1,1,1,1]
#     weight = torch.FloatTensor(weight).to(truth.device).view(1,-1)
#
#     loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
#     loss = loss*weight
#     loss = loss.mean()
#     return loss


# https://discuss.pytorch.org/t/numerical-stability-of-bcewithlogitsloss/8246
def criterion_attention(logit, truth, weight=None):
    batch_size, num_class, H, W = logit.shape

    if weight is None: weight = [1, 1, 1, 1]
    weight = torch.FloatTensor(weight).to(truth.device).view(1, -1, 1, 1)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    # ---
    # https://arxiv.org/pdf/1909.07829.pdf
    if 0:
        gamma = 2.0
        p = torch.sigmoid(logit)
        focal = (truth * (1 - p) + (1 - truth) * (p)) ** gamma
        weight = weight * focal / focal.sum().item() * H * W
    # ---
    loss = loss * weight
    loss = loss.mean()
    return loss


#
# def criterion_mask(logit, truth, weight=None):
#     if weight is not None: weight = torch.FloatTensor([1]+weight).cuda()
#     batch_size,num_class,H,W = logit.shape
#
#     logit = logit.permute(0, 2, 3, 1).contiguous().view(batch_size,-1, 5)
#     log_probability = -F.log_softmax(logit,-1)
#
#
#     truth = truth.permute(0, 2, 3, 1).contiguous().view(-1,1)
#     onehot = torch.FloatTensor(batch_size*H*W, 5).to(truth.device)
#     onehot.zero_()
#     onehot.scatter_(1, truth, 1)
#     onehot = onehot.view(batch_size,-1, 5)
#
#     #loss = F.cross_entropy(logit, truth, weight=weight, reduction='none')
#     loss = log_probability*onehot
#
#     loss = loss*weight
#     loss = loss.mean()
#     return loss

# focal loss
def criterion_mask(logit, truth, weight=None):
    if weight is None: weight = [1, 1, 1, 1]
    weight = torch.FloatTensor([1] + weight).to(truth.device).view(1, -1)

    batch_size, num_class, H, W = logit.shape

    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
    truth = truth.permute(0, 2, 3, 1).contiguous().view(-1)
    # return F.cross_entropy(logit, truth, reduction='mean')

    log_probability = -F.log_softmax(logit, -1)
    probability = F.softmax(logit, -1)

    onehot = torch.zeros(batch_size * H * W, num_class).to(truth.device)
    onehot.scatter_(dim=1, index=truth.view(-1, 1), value=1)  # F.one_hot(truth,5).float()

    loss = log_probability * onehot

    # ---
    if 1:  # image based focusing
        probability = probability.view(batch_size, H * W, 5)
        truth = truth.view(batch_size, H * W, 1)
        weight = weight.view(1, 1, 5)

        alpha = 2
        focal = torch.gather(probability, dim=-1, index=truth.view(batch_size, H * W, 1))
        focal = (1 - focal) ** alpha
        focal_sum = focal.sum(dim=[1, 2], keepdim=True)
        # focal_sum = focal.sum().view(1,1,1)
        weight = weight * focal / focal_sum.detach() * H * W
        weight = weight.view(-1, 5)

    loss = loss * weight
    loss = loss.mean()
    return loss


# ----
def logit_mask_to_probability_label(logit):
    batch_size, num_class, H, W = logit.shape
    probability = F.softmax(logit, 1)
    # probability = F.avg_pool2d(probability, kernel_size=16,stride=16)

    probability = probability.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 5)
    value, index = probability.max(1)

    probability = value[:, 1:]
    return probability


def metric_label(probability, truth, threshold=0.5):
    batch_size = len(truth)

    with torch.no_grad():
        probability = probability.view(batch_size, 4)
        truth = truth.view(batch_size, 4)

        # ----
        neg_index = (truth == 0).float()
        pos_index = 1 - neg_index
        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)

        # ----
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives
        tn = tn.sum(0)
        tp = tp.sum(0)

        # ----
        tn = tn.data.cpu().numpy()
        tp = tp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return tn, tp, num_neg, num_pos


def truth_to_onehot(truth, num_class=4):
    onehot = truth.repeat(1, num_class, 1, 1)
    arange = torch.arange(1, num_class + 1).view(1, num_class, 1, 1).to(truth.device)
    onehot = (onehot == arange).float()
    return onehot


def predict_to_onehot(predict, num_class=4):
    value, index = torch.max(predict, 1, keepdim=True)
    value = value.repeat(1, num_class, 1, 1)
    index = index.repeat(1, num_class, 1, 1)
    arange = torch.arange(1, num_class + 1).view(1, num_class, 1, 1).to(predict.device)
    onehot = (index == arange).float()
    value = value * onehot
    return value


def metric_mask(logit, truth, threshold=0.5, sum_threshold=100):
    with torch.no_grad():
        probability = torch.softmax(logit, 1)
        truth = truth_to_onehot(truth)
        probability = predict_to_onehot(probability)

        batch_size, num_class, H, W = truth.shape
        probability = probability.view(batch_size, num_class, -1)
        truth = truth.view(batch_size, num_class, -1)
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        d_neg = (p_sum < sum_threshold).float()
        d_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1) + 1e-12)

        neg_index = (t_sum == 0).float()
        pos_index = 1 - neg_index

        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)
        dn = (neg_index * d_neg).sum(0)
        dp = (pos_index * d_pos).sum(0)

        # ----
        dn = dn.data.cpu().numpy()
        dp = dp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return dn, dp, num_neg, num_pos


##############################################################################################
def make_dummy_data(batch_size=8):
    image_id = np.array([
        i + '.jpg' for i in [
            '0a8fddf7a', '0a29ef6f9', '0a46cc4bf', '0a058fcb6', '0a65bd8d4', '0a427a066', '0a6324223', '0b89f99d7',
            '00ac8372f', '1ae56dead', '1b7bec2ba', '1bdb7f26f', '1cac6e1f3', '1d34ad26c', '1d83b44be', '1e75373b2',
            '0b4c8e681', '0b5018316', '2b01fd731', '0cb590f8e', '0d4866e3c', '0e106d482', '0ebdc1277', '1bed9264f',
            '0a9aaba9a', '0a26aceb2', '0a405b396', '0aa7955fd', '0bda9a0eb', '0c2522533', '0cd22bad5', '0ce3a145f',
            '0adc17f1d', '0b56da4ff', '0be9bad7b', '0c888ecb5', '0d4eae8de', '0d78ac743', '0d51538b9', '0ddbc9fb5',
        ]
    ]).reshape(5, -1).T.reshape(-1).tolist()

    DATA_DIR = '/root/share/project/kaggle/2019/steel/data'
    folder = 'train_images'

    df = pd.read_csv(DATA_DIR + '/train.csv').fillna('')
    df = df_loc_by_list(df, 'ImageId_ClassId', [i + '_%d' % c for i in image_id for c in [1, 2, 3, 4]])
    df = df.reset_index(drop=True)
    # print(df)
    # exit(0)

    batch = []
    for b in range(0, batch_size):
        num_image = len(df) // 4
        i = b % num_image

        image_id = df['ImageId_ClassId'].values[i * 4][:-2]
        rle = df['EncodedPixels'].values[i * 4:(i + 1) * 4:]
        image = cv2.imread(DATA_DIR + '/%s/%s' % (folder, image_id), cv2.IMREAD_COLOR)
        label = [0 if r == '' else 1 for r in rle]
        mask = np.array([run_length_decode(r, height=256, width=1600, fill_value=c) for c, r in zip([1, 2, 3, 4], rle)])

        # ---
        # crop to 256x400
        w = 400
        mask_sum = mask.sum(1).sum(0)
        mask_sum = mask_sum.cumsum()
        mask_sum = mask_sum[w:] - mask_sum[:-w]
        x = np.argmax(mask_sum)
        image = image[:, x:x + w]
        mask = mask[:, :, x:x + w]

        zz = 0
        # ---

        mask = mask.max(0, keepdims=0)
        infor = Struct(
            index=i,
            folder=folder,
            image_id=image_id,
        )

        batch.append([image, label, mask, infor])

    input, truth_label, truth_mask, truth_attention, infor = null_collate(batch)
    input = input.cuda()
    truth_label = truth_label.cuda()
    truth_mask = truth_mask.cuda()
    truth_attention = truth_attention.cuda()

    return input, truth_label, truth_mask, truth_attention, infor


#########################################################################
def run_check_basenet():
    net = Net()
    print(net)
    net.load_pretrain(skip=['logit'])


def run_check_net():
    batch_size = 1
    C, H, W = 3, 256, 400
    num_class = 4

    input = np.random.uniform(-1, 1, (batch_size, C, H, W))
    input = np.random.uniform(-1, 1, (batch_size, C, H, W))
    input = torch.from_numpy(input).float().cuda()

    net = Net(num_class=num_class).cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('')
    print('input: ', input.shape)
    print('logit: ', logit.shape)
    # print(net)


def run_check_train():
    loss_weight = [1, 1, 1, 1]
    if 1:
        input, truth_label, truth_mask, truth_attention, infor = make_dummy_data(batch_size=10)
        batch_size, C, H, W = input.shape

        print('input: ', input.shape)
        print('truth_label: ', truth_label.shape)
        print('(count)    : ', truth_label.sum(0))
        print('truth_mask: ', truth_mask.shape)
        print('truth_attention: ', truth_attention.shape)
        print('')

    # ---

    net = Net().cuda()
    net.load_pretrain(is_print=False)  #

    net = net.eval()
    with torch.no_grad():
        logit_mask = net(input)
        print('input: ', input.shape)
        print('logit_mask: ', logit_mask.shape)
        print('')

        loss = criterion_mask(logit_mask, truth_mask, loss_weight)
        probability_label = logit_mask_to_probability_label(logit_mask)
        tn, tp, num_neg, num_pos = metric_label(probability_label, truth_label)
        dn, dp, num_neg, num_pos = metric_mask(logit_mask, truth_mask)

        print('loss = %0.5f' % loss.item())
        print('tn,tp = [%0.3f,%0.3f,%0.3f,%0.3f], [%0.3f,%0.3f,%0.3f,%0.3f] ' % (*(tn / num_neg), *(tp / num_pos)))
        print('tn,tp = [%0.3f,%0.3f,%0.3f,%0.3f], [%0.3f,%0.3f,%0.3f,%0.3f] ' % (*(dn / num_neg), *(dp / num_pos)))
        print('num_pos,num_neg = [%d,%d,%d,%d], [%d,%d,%d,%d] ' % (*num_neg, *num_pos))
        print('')

    # exit(0)
    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0001)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =', batch_size)
    print('---------------------------------------------------------------------------------------------------------')
    print('[iter ]  loss              |    tn, [tp1,tp2,tp3,tp4]          |    dn, [dp1,dp2,dp3,dp4]   ')
    print('---------------------------------------------------------------------------------------------------------')
    # [00000]  1.91935, 0.27055  | 0.533, [1.000,0.500,0.000,0.000]  | 0.000, [0.003,0.016,0.102,0.073]

    i = 0
    optimizer.zero_grad()
    while i <= 150:

        net.train()
        optimizer.zero_grad()

        logit_mask = net(input)
        loss = criterion_mask(logit_mask, truth_mask, loss_weight)
        probability_label = logit_mask_to_probability_label(logit_mask)
        tn, tp, num_neg, num_pos = metric_label(probability_label, truth_label)
        dn, dp, num_neg, num_pos = metric_mask(logit_mask, truth_mask)

        (loss).backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                '[%05d] %8.5f | [%0.2f,%0.2f,%0.2f,%0.2f], [%0.2f,%0.2f,%0.2f,%0.2f] | [%0.2f,%0.2f,%0.2f,%0.2f], [%0.2f,%0.2f,%0.2f,%0.2f] ' % (
                    i,
                    loss.item(),
                    *(tn / num_neg), *(tp / num_pos),
                    *(dn / num_neg), *(dp / num_pos),
                ))
        i = i + 1
    print('')

    # exit(0)
    if 1:
        # net.eval()
        logit_mask = net(input)
        probability_label = logit_mask_to_probability_label(logit_mask)
        probability_mask = F.softmax(logit_mask, 1)

        probability_label = probability_label.data.cpu().numpy()
        probability_mask = probability_mask.data.cpu().numpy()
        truth_label = truth_label.data.cpu().numpy()
        truth_mask = truth_mask.data.cpu().numpy()

        image = input_to_image(input)
        for b in range(batch_size):
            print('%2d ------ ' % (b))
            result = draw_predict_result(
                image[b], truth_label[b], truth_mask[b], probability_label[b], probability_mask[b])

            image_show('result', result, resize=0.5)
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # run_check_basenet()
    # run_check_net()
    run_check_train()

    print('\nsucess!')