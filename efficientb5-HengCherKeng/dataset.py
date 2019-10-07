from common import *
from etc import *


DATA_DIR = '/root/share/project/kaggle/2019/steel/data'


class SteelDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):

        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.augment = augment

        self.uid = list(np.concatenate([np.load(DATA_DIR + '/split/%s'%f , allow_pickle=True) for f in split]))
        df = pd.concat([pd.read_csv(DATA_DIR + '/%s'%f).fillna('') for f in csv])

        df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
        df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
        df = df_loc_by_list(df, 'ImageId_ClassId', [ u.split('/')[-1] + '_%d'%c  for u in self.uid for c in [1,2,3,4] ])
        self.df = df
        self.num_image = len(df)//4


    def __str__(self):
        num1 = (self.df['Class']==1).sum()
        num2 = (self.df['Class']==2).sum()
        num3 = (self.df['Class']==3).sum()
        num4 = (self.df['Class']==4).sum()
        pos1 = ((self.df['Class']==1) & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']==2) & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']==3) & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']==4) & (self.df['Label']==1)).sum()
        neg1 = num1-pos1
        neg2 = num2-pos2
        neg3 = num3-pos3
        neg4 = num4-pos4

        length = len(self)
        num = len(self)
        pos = (self.df['Label']==1).sum()
        neg = num-pos

        #---

        string  = ''
        string += '\tmode    = %s\n'%self.mode
        string += '\tsplit   = %s\n'%self.split
        string += '\tcsv     = %s\n'%str(self.csv)
        string += '\tnum_image = %8d\n'%self.num_image
        string += '\tlen       = %8d\n'%len(self)
        if self.mode == 'train':
            string += '\t\tpos1, neg1 = %5d  %0.3f,  %5d  %0.3f\n'%(pos1,pos1/num,neg1,neg1/num)
            string += '\t\tpos2, neg2 = %5d  %0.3f,  %5d  %0.3f\n'%(pos2,pos2/num,neg2,neg2/num)
            string += '\t\tpos3, neg3 = %5d  %0.3f,  %5d  %0.3f\n'%(pos3,pos3/num,neg3,neg3/num)
            string += '\t\tpos4, neg4 = %5d  %0.3f,  %5d  %0.3f\n'%(pos4,pos4/num,neg4,neg4/num)
        return string


    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        # print(index)
        folder, image_id = self.uid[index].split('/')

        rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread(DATA_DIR + '/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)
        label = [ 0 if r=='' else 1 for r in rle]
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=c) for c,r in zip([1,2,3,4],rle)])
        mask  = mask.max(0, keepdims=0)

        infor = Struct(
            index    = index,
            folder   = folder,
            image_id = image_id,
        )

        if self.augment is None:
            return image, label, mask, infor
        else:
            return self.augment(image, label, mask, infor)



def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth_label = []
    truth_mask  = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth_label.append(batch[b][1])
        truth_mask.append(batch[b][2])
        infor.append(batch[b][3])

    input = np.stack(input).astype(np.float32)/255
    input = input.transpose(0,3,1,2)
    truth_label = np.stack(truth_label)
    truth_mask  = np.stack(truth_mask)

    input = torch.from_numpy(input).float()
    truth_label = torch.from_numpy(truth_label).float()
    truth_mask = torch.from_numpy(truth_mask).long().unsqueeze(1)

    return input, truth_label, truth_mask, infor


##############################################################

class FiveBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['Label'].values)
        label = label.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        #5x
        self.num_image = len(self.dataset.df)//4
        self.length = self.num_image*5


    def __iter__(self):
        # neg = self.neg_index.copy()
        # random.shuffle(neg)

        neg  = np.random.choice(self.neg_index,  self.num_image, replace=True)
        pos1 = np.random.choice(self.pos1_index, self.num_image, replace=True)
        pos2 = np.random.choice(self.pos2_index, self.num_image, replace=True)
        pos3 = np.random.choice(self.pos3_index, self.num_image, replace=True)
        pos4 = np.random.choice(self.pos4_index, self.num_image, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


class FixedSampler(Sampler):

    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index   = index
        self.length  = len(index)

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return self.length

class FixedRandomSampler(Sampler):

    def __init__(self, dataset, length=-1):
        self.dataset = dataset

        if length<0: length = len(dataset)
        self.length = length

    def __iter__(self):
        L = len(self.dataset)
        index  = np.random.choice(L,  self.length, replace=True)
        return iter(index)

    def __len__(self):
        return self.length

##############################################################

#def image_to_input(image,rbg_mean,rbg_std):
def image_to_input(image,rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    input[:,0] = (input[:,0]-rbg_mean[0])/rbg_std[0]
    input[:,1] = (input[:,1]-rbg_mean[1])/rbg_std[1]
    input[:,2] = (input[:,2]-rbg_mean[2])/rbg_std[2]
    return input

#def input_to_image(input,rbg_mean,rbg_std)
def input_to_image(input, rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = input.data.cpu().numpy()
    input[:,0] = (input[:,0]*rbg_std[0]+rbg_mean[0])
    input[:,1] = (input[:,1]*rbg_std[1]+rbg_mean[1])
    input[:,2] = (input[:,2]*rbg_std[2]+rbg_mean[2])
    input = input.transpose(0,2,3,1)
    input = input[...,::-1]
    image = (input*255).astype(np.uint8)
    return image



##############################################################

def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask  = mask[:,::-1]
    return image, mask

def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask  = mask[::-1,:]
    return image, mask

def do_random_crop(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [y:y+h,x:x+w]
    return image, mask

def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w]
    mask  = mask [y:y+h,x:x+w]

    #---
    if (w,h)!=(width,height):
        image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize( mask,  dsize=(width,height), interpolation=cv2.INTER_NEAREST)

    return image, mask

def do_random_crop_rotate_rescale(image, mask, w, h):
    H,W = image.shape[:2]

    #dangle = np.random.uniform(-2.5, 2.5)
    #dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-8, 8)
    dshift = np.random.uniform(-0.1,0.1,2)

    dscale_x = np.random.uniform(-0.00075,0.00075)
    dscale_y = np.random.uniform(-0.25,0.25)

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx,sy = 1 + dscale_x, 1+ dscale_y #1,1 #
    tx,ty = dshift*min(H,W)

    src = np.array([[-w/2,-h/2],[ w/2,-h/2],[ w/2, h/2],[-w/2, h/2]], np.float32)
    src = src*[sx,sy]
    x = (src*[cos,-sin]).sum(1)+W/2
    y = (src*[sin, cos]).sum(1)+H/2
    # x = x-x.min()
    # y = y-y.min()
    # x = x + (W-x.max())*tx
    # y = y + (H-y.max())*ty

    if 0:
        overlay=image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i],y[i]]), int_tuple([x[(i+1)%4],y[(i+1)%4]]), (0,0,255),5)
        image_show('overlay',overlay)
        cv2.waitKey(0)


    src = np.column_stack([x,y])
    dst = np.array([[0,0],[w,0],[w,h],[0,h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s,d)

    image = cv2.warpPerspective( image, transform, (W, H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    mask = cv2.warpPerspective( mask, transform, (W, H),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    return image, mask

def do_random_log_contast(image, gain=[0.70, 1.30] ):
    gain = np.random.uniform(gain[0],gain[1],1)
    inverse = np.random.choice(2,1)

    image = image.astype(np.float32)/255
    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255,0,255).astype(np.uint8)
    return image

def do_random_noise(image, noise=8):
    H,W = image.shape[:2]
    image = image.astype(np.float32)
    image = image + np.random.uniform(-1,1,(H,W,1))*noise
    image = np.clip(image,0,255).astype(np.uint8)
    return image

##---
#https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
def do_random_contast(image):
    beta=0
    alpha=random.uniform(0.5, 2.0)

    image = image.astype(np.float32) * alpha + beta
    image = np.clip(image,0,255).astype(np.uint8)
    return image

#----
## customize
def do_random_salt_pepper_noise(image, noise =0.0005):
    height,width = image.shape[:2]
    num_salt = int(noise*width*height)

    # Salt mode
    yx = [np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
    image[tuple(yx)] = [255,255,255]

    # Pepper mode
    yx = [np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
    image[tuple(yx)] = [0,0,0]

    return image



def do_random_salt_pepper_line(image, noise =0.0005, length=10):
    height,width = image.shape[:2]
    num_salt = int(noise*width*height)

    # Salt mode
    y0x0 = np.array([np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]).T
    y1x1 = y0x0 + np.random.choice(2*length, size=(num_salt,2))-length
    for (y0,x0), (y1,x1)  in zip(y0x0,y1x1):
        cv2.line(image,(x0,y0),(x1,y1), (255,255,255), 1)

    # Pepper mode
    y0x0 = np.array([np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]).T
    y1x1 = y0x0 + np.random.choice(2*length, size=(num_salt,2))-length
    for (y0,x0), (y1,x1)  in zip(y0x0,y1x1):
        cv2.line(image,(x0,y0),(x1,y1), (0,0,0), 1)

    return image


def do_random_cutout(image, mask):
    height, width = image.shape[:2]

    u0 = [0,1][np.random.choice(2)]
    u1 = np.random.choice(width)

    if u0 ==0:
        x0,x1=0,u1
    if u0 ==1:
        x0,x1=u1,width

    image[:,x0:x1]=0
    mask [:,x0:x1]=0
    return image,mask



# def do_random_special1(image, mask):
#     height, width = image.shape[:2]
#
#     if np.random.rand()>0.5:
#         y = np.random.choice(height)
#         image = np.vstack([image[y:],image[:y]])
#         mask = np.vstack([mask[y:],mask[:y]])
#
#     if np.random.rand()>0.5:
#         x = np.random.choice(width)
#         image = np.hstack([image[:,x:],image[:,:x]])
#         mask = np.hstack([mask[:,x:],mask[:,:x]])
#
#     return image,mask

##############################################################

def run_check_train_dataset():

    # dataset = SteelDataset(
    #     mode    = 'test',
    #     csv     = ['sample_submission.csv',],
    #     split   = ['test_1801.npy',],
    #     augment = None, #
    # )

    dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_b0_11568.npy',],
        augment = None, #
    )
    print(dataset)
    #exit(0)

    for n in range(0,len(dataset)):
        i = n #i = np.random.choice(len(dataset))

        image, label, mask, infor = dataset[i]
        overlay = draw_truth_mask_overlay(image, mask)

        #----
        print('%05d : %s'%(i, infor.image_id))
        print('label = %s'%str(label))
        print('')
        image_show('image',image,0.5)
        image_show_norm('mask',mask,0,4,0.5)
        image_show('overlay',overlay,0.5)
        cv2.waitKey(0)




def run_check_data_loader():

    dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_b0_11568.npy',],
        augment = None, #
    )
    print(dataset)
    loader  = DataLoader(
        dataset,
        sampler     = FiveBalanceClassSampler(dataset),
        #sampler     = SequentialSampler(dataset),
        #sampler     = RandomSampler(dataset),
        batch_size  = 5,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    for t,(input, truth_label, truth_mask, infor) in enumerate(loader):

        print('----t=%d---'%t)
        print('')
        print(infor)
        print('input', input.shape)
        print('truth_label', truth_label.shape)
        print('truth_mask ', truth_mask.shape)
        print('')

        if 1:
            batch_size= len(infor)
            input = input.data.cpu().numpy()
            input = (input*255).astype(np.uint8)
            input = input.transpose(0,2,3,1)
            #input = 255-(input*255).astype(np.uint8)

            truth_label = truth_label.data.cpu().numpy()
            truth_mask = truth_mask.data.cpu().numpy()
            for b in range(batch_size):
                print(infor[b].image_id)

                image = input[b]
                label = truth_label[b]
                mask  = truth_mask[b,0]
                overlay = draw_truth_mask_overlay(image, mask)

                print('%05d : %s'%(b, infor[b].image_id))
                print('label = %s'%str(label))
                print('')
                image_show('image',image,0.5)
                image_show_norm('mask',mask,0,4,0.5)
                image_show('overlay',overlay,0.5)
                cv2.waitKey(0)



def run_check_augment():

    def augment(image, label, mask, infor):
        #if np.random.rand()<0.5: image, mask = do_flip_ud(image, mask)
        #if np.random.rand()<0.5: image, mask = do_flip_lr(image, mask)

        #image, mask = do_random_crop_rescale(image,mask,1600-(256-180),220)
        #image, mask = do_random_crop_rotate_rescale(image,mask,1600-(256-224),224)
        #image = do_random_log_contast(image, gain=[0.70, 1.50])

        #image = do_random_special0(image)

        #image, mask = do_random_special0(image, mask)


        #image, mask = do_random_cutout(image, mask)
        #image = do_random_salt_pepper_line(image, noise =0.0005, length=10)
        image = do_random_salt_pepper_noise(image, noise =0.005)
        #image = do_random_log_contast(image, gain=[0.50, 1.75])

        # a,b,c = np.random.uniform(-0.25,0.25,3)
        # augment = image.astype(np.float32)/255
        # augment = (augment-0.5+a)*(1+b*2) + (0.5+c)
        # image = np.clip(augment*255,0,255).astype(np.uint8)
        #
        #
        # image = do_random_noise(image, noise=16)

        return image, label, mask, infor


    dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_b0_11568.npy',],
        augment = None,  #None
    )
    print(dataset)


    for b in range(len(dataset)):
        image, label, mask, infor = dataset[b]
        overlay = draw_truth_mask_overlay(image.copy(), mask)
        result = np.vstack([image,overlay])
        result = draw_grid(result,(1600,256),(255,255,255),1)

        #---
        #dilation = cv2.dilate(img,kernel,iterations = 1)


        #---


        print('----b=%d---'%b)
        print('')
        print('infor\n',infor)
        print(image.shape)
        print(mask.shape)
        print('')

        image_show('before',result,resize=0.5)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, label1, mask1, infor =  augment(image.copy(), label.copy(), mask.copy(), infor)
                overlay1 = draw_truth_mask_overlay(image1.copy(), mask1)
                result1 = np.vstack([image1,overlay1])
                result1 = draw_grid(result1,(1600,256),(255,255,255),1)


                image_show('after',result1,resize=0.5)
                cv2.waitKey(0)


def run_check_batch_augment():

    dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train_b0_11568.npy',],
        augment = None,  #None
    )
    print(dataset)
    loader  = DataLoader(
        dataset,
        sampler     = FiveBalanceClassSampler(dataset),
        #sampler     = SequentialSampler(dataset),
        #sampler     = RandomSampler(dataset),
        batch_size  = 5,
        drop_last   = False,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate
    )

    def do_cutmix(input, truth):
        batch_size = len(truth)
        perm = torch.randperm(batch_size).to(input.device)
        perm_input = input[perm]
        perm_truth = truth[perm]

        select = []
        for b in range(batch_size):
            width = 1600//32
            w = int(np.random.uniform(0.25,0.50)*width)
            x = np.random.choice(width-w)
            w = int(w*32)
            x = int(x*32)

            s = np.zeros((256,1600), np.float32)
            s[:,x:x+w] = 1
            select.append(s)

            image_show_norm('s',s)
            cv2.waitKey(1)
        select = np.array(select)
        select = torch.from_numpy(select).to(input.device).unsqueeze(1)

        select_input = select.repeat(1,3,1,1)
        input = (1-select_input)*input + select_input*perm_input

        select_truth = F.interpolate(select,size=truth.shape[-2:],mode='nearest')
        select_truth = select_truth.repeat(1,1,1,1).long()
        truth = (1-select_truth)*truth + select_truth*perm_truth


        if 1: #debug only
            select = select.data.cpu().numpy()
            perm_input  = perm_input.data.cpu().numpy()
            input  = input.data.cpu().numpy()

            truth  = truth.data.cpu().numpy()

            for b in range(batch_size):
                s = select[b,0]
                perm_m = perm_input[b,0]
                m = input[b,0]
                t = truth[b,0]
                image_show_norm('s',s)
                image_show_norm('perm_m',perm_m)
                image_show_norm('m',m)
                image_show_norm('t',t)
                cv2.waitKey(0)

        return input, truth


    for t,(input, truth_label, truth_mask, infor) in enumerate(loader):

        input1, truth1 = do_cutmix(input, truth_mask)


        print('----t=%d---'%t)
        print('')
        print(infor)
        print('input', input.shape)
        print('truth_label', truth_label.shape)
        print('truth_mask ', truth_mask.shape)
        print('')

        if 0:
            batch_size = len(infor)
            input = input.data.cpu().numpy()
            input = (input*255).astype(np.uint8)
            input = input.transpose(0,2,3,1)
            #input = 255-(input*255).astype(np.uint8)

            truth_label = truth_label.data.cpu().numpy()
            truth_mask = truth_mask.data.cpu().numpy()
            for b in range(batch_size):
                print(infor[b].image_id)

                image = input[b]
                label = truth_label[b]
                mask  = truth_mask[b,0]
                overlay = draw_truth_mask_overlay(image, mask)

                print('%05d : %s'%(b, infor[b].image_id))
                print('label = %s'%str(label))
                print('')
                image_show('image',image,0.5)
                image_show('overlay',overlay,0.5)
                image_show_norm('mask',mask,0,4,0.5)
                cv2.waitKey(0)
            #---


##----

def run_etc0():

    file = glob.glob('/root/share/project/kaggle/2019/steel/data/train_images/*.jpg')
    train_id = [f.split('/')[-1][:-4]+'.jpg' for f in file]
    for image_id in train_id:
        print(image_id)

        folder = 'train_images'
        image = cv2.imread(DATA_DIR + '/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)

        m = image.mean(-1)
        m = m.mean(0, keepdims=True)
        m = (m>24).astype(np.float32)

        select = np.zeros((256,1600), np.uint8)
        select.fill(255)
        select = select*m
        select = cv2.GaussianBlur(select,(41,41),0)


        image_show('image',image)
        image_show('select',select)
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    #run_check_train_dataset()
    #run_check_data_loader()
    run_check_augment()
    #run_check_batch_augment()

    #run_etc0()


