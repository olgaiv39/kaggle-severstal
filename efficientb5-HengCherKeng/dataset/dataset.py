import torch
import numpy
import pandas
import cv2
from .. import etc


class SteelDataset(torch.utils.data.Dataset):
    def __init__(self, split, csv, mode, data_dir, augment=None):
        self.split = split
        self.csv = csv
        self.mode = mode
        self.data_dir = data_dir
        self.augment = augment
        self.uid = list(
            numpy.concatenate(
                [numpy.load(data_dir + "/split/%s" % f, allow_pickle=True) for f in split]
            )
        )
        dataframe = pandas.concat([pandas.read_csv(data_dir + "/%s" % f).fillna("") for f in csv])
        dataframe["Class"] = dataframe["ImageId_ClassId"].str[-1].astype(numpy.int32)
        dataframe["Label"] = (dataframe["EncodedPixels"] != "").astype(numpy.int32)
        # TODO: Implement dataframe_loc_by_list
        dataframe = dataframe_loc_by_list(
            dataframe,
            "ImageId_ClassId",
            [u.split("/")[-1] + "_%d" % c for u in self.uid for c in [1, 2, 3, 4]],
        )
        self.dataframe = dataframe
        self.num_image = len(dataframe) // 4

    def __str__(self):
        num1 = (self.dataframe["Class"] == 1).sum()
        num2 = (self.dataframe["Class"] == 2).sum()
        num3 = (self.dataframe["Class"] == 3).sum()
        num4 = (self.dataframe["Class"] == 4).sum()
        pos1 = ((self.dataframe["Class"] == 1) & (self.dataframe["Label"] == 1)).sum()
        pos2 = ((self.dataframe["Class"] == 2) & (self.dataframe["Label"] == 1)).sum()
        pos3 = ((self.dataframe["Class"] == 3) & (self.dataframe["Label"] == 1)).sum()
        pos4 = ((self.dataframe["Class"] == 4) & (self.dataframe["Label"] == 1)).sum()
        neg1 = num1 - pos1
        neg2 = num2 - pos2
        neg3 = num3 - pos3
        neg4 = num4 - pos4
        length = len(self)
        num = len(self)
        pos = (self.dataframe["Label"] == 1).sum()
        neg = num - pos
        string = ""
        string += "\tmode    = %s\n" % self.mode
        string += "\tsplit   = %s\n" % self.split
        string += "\tcsv     = %s\n" % str(self.csv)
        string += "\tnum_image = %8d\n" % self.num_image
        string += "\tlen       = %8d\n" % len(self)
        if self.mode == "train":
            string += "\t\tpos1, neg1 = %5d  %0.3f,  %5d  %0.3f\n" % (
                pos1,
                pos1 / num,
                neg1,
                neg1 / num,
            )
            string += "\t\tpos2, neg2 = %5d  %0.3f,  %5d  %0.3f\n" % (
                pos2,
                pos2 / num,
                neg2,
                neg2 / num,
            )
            string += "\t\tpos3, neg3 = %5d  %0.3f,  %5d  %0.3f\n" % (
                pos3,
                pos3 / num,
                neg3,
                neg3 / num,
            )
            string += "\t\tpos4, neg4 = %5d  %0.3f,  %5d  %0.3f\n" % (
                pos4,
                pos4 / num,
                neg4,
                neg4 / num,
            )
        return string

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        folder, image_id = self.uid[index].split("/")
        rle = [
            self.dataframe.loc[
                self.dataframe["ImageId_ClassId"] == image_id + "_1", "EncodedPixels"
            ].values[0],
            self.dataframe.loc[
                self.dataframe["ImageId_ClassId"] == image_id + "_2", "EncodedPixels"
            ].values[0],
            self.dataframe.loc[
                self.dataframe["ImageId_ClassId"] == image_id + "_3", "EncodedPixels"
            ].values[0],
            self.dataframe.loc[
                self.dataframe["ImageId_ClassId"] == image_id + "_4", "EncodedPixels"
            ].values[0],
        ]
        image = cv2.imread(self.data_dir + "/%s/%s" % (folder, image_id), cv2.IMREAD_COLOR)
        label = [0 if r == "" else 1 for r in rle]
        mask = numpy.array(
            [
                etc.run_length_decode(r, height=256, width=1600, fill_value=c)
                for c, r in zip([1, 2, 3, 4], rle)
            ]
        )
        mask = mask.max(0, keepdims=0)
        infor = Struct(index=index, folder=folder, image_id=image_id)
        if self.augment is None:
            return image, label, mask, infor
        else:
            return self.augment(image, label, mask, infor)
