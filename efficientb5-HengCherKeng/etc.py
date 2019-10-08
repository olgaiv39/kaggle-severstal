from common import *

from sklearn import metrics as sklearn_metrics


## kaggle ###########################################################################

# https://www.kaggle.com/iafoss/severstal-fast-ai-256x256-crops-sub
# https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88


def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height, width), np.float32)
    if rle != "":
        mask = mask.reshape(-1)
        r = [int(r) for r in rle.split(" ")]
        r = np.array(r).reshape(-1, 2)
        for start, length in r:
            start = start - 1  # ???? 0 or 1 index ???
            mask[start : (start + length)] = fill_value
        mask = mask.reshape(width, height).T
    return mask


def run_length_encode(mask):
    # possible bug for here
    m = mask.T.flatten()
    if m.sum() == 0:
        rle = ""
    else:
        m = np.concatenate([[0], m, [0]])
        run = np.where(m[1:] != m[:-1])[0] + 1
        run[1::2] -= run[::2]
        rle = " ".join(str(r) for r in run)
    return rle


## kaggle data #####################################################################


# https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/107053#latest-617549
DUPLICATE = (
    np.array(
        [
            "train_images/6eb8690cd.jpg",
            "train_images/a67df9196.jpg",
            "train_images/24e125a16.jpg",
            "train_images/4a80680e5.jpg",
            "train_images/a335fc5cc.jpg",
            "train_images/fb352c185.jpg",
            "train_images/c35fa49e2.jpg",
            "train_images/e4da37c1e.jpg",
            "train_images/877d319fd.jpg",
            "train_images/e6042b9a7.jpg",
            "train_images/618f0ff16.jpg",
            "train_images/ace59105f.jpg",
            "train_images/ae35b6067.jpg",
            "train_images/fdb5ae9d4.jpg",
            "train_images/3de8f5d88.jpg",
            "train_images/a5aa4829b.jpg",
            "train_images/3bd0fd84d.jpg",
            "train_images/b719010ac.jpg",
            "train_images/24fce7ae0.jpg",
            "train_images/edf12f5f1.jpg",
            "train_images/49e374bd3.jpg",
            "train_images/6099f39dc.jpg",
            "train_images/9b2ed195e.jpg",
            "train_images/c30ecf35c.jpg",
            "train_images/3a7f1857b.jpg",
            "train_images/c37633c03.jpg",
            "train_images/8c2a5c8f7.jpg",
            "train_images/abedd15e2.jpg",
            "train_images/b46dafae2.jpg",
            "train_images/ce5f0cec3.jpg",
            "train_images/5b1c96f09.jpg",
            "train_images/e054a983d.jpg",
            "train_images/3088a6a0d.jpg",
            "train_images/7f3181e44.jpg",
            "train_images/dc0c6c0de.jpg",
            "train_images/e4d9efbaa.jpg",
            "train_images/488c35cf9.jpg",
            "train_images/845935465.jpg",
            "train_images/3b168b16e.jpg",
            "train_images/c6af2acac.jpg",
            "train_images/05bc27672.jpg",
            "train_images/dfefd11c4.jpg",
            "train_images/048d14d3f.jpg",
            "train_images/7c8a469a4.jpg",
            "train_images/a1a0111dd.jpg",
            "train_images/b30a3e3b6.jpg",
            "train_images/d8be02bfa.jpg",
            "train_images/e45010a6a.jpg",
            "train_images/caf49d870.jpg",
            "train_images/ef5c1b08e.jpg",
            "train_images/63c219c6f.jpg",
            "train_images/b1096a78f.jpg",
            "train_images/76096b17b.jpg",
            "train_images/d490180a3.jpg",
            "train_images/bd0e26062.jpg",
            "train_images/e7d7c87e2.jpg",
            "train_images/600a81590.jpg",
            "train_images/eb5aec756.jpg",
            "train_images/ad5a2ea44.jpg",
            "train_images/e9fa75516.jpg",
            "train_images/6afa917f2.jpg",
            "train_images/9fb53a74b.jpg",
            "train_images/59931eb56.jpg",
            "train_images/e7ced5b76.jpg",
            "train_images/0bfe252d0.jpg",
            "train_images/b4d0843ed.jpg",
            "train_images/67fc6eeb8.jpg",
            "train_images/c04aa9618.jpg",
            "train_images/741a5c461.jpg",
            "train_images/dae3c563a.jpg",
            "train_images/78416c3d0.jpg",
            "train_images/e34f68168.jpg",
            "train_images/0d258e4ae.jpg",
            "train_images/72322fc23.jpg",
            "train_images/0aafd7471.jpg",
            "train_images/461f83c57.jpg",
            "train_images/38a1d7aab.jpg",
            "train_images/8866a93f6.jpg",
            "train_images/7c5b834b7.jpg",
            "train_images/dea514023.jpg",
            "train_images/32854e5bf.jpg",
            "train_images/530227cd2.jpg",
            "train_images/1b7d7eec6.jpg",
            "train_images/f801dd10b.jpg",
            "train_images/46ace1c15.jpg",
            "train_images/876e74fd6.jpg",
            "train_images/578b43574.jpg",
            "train_images/9c5884cdd.jpg",
        ]
    )
    .reshape(-1, 2)
    .tolist()
)


## metric  ###########################################################################


def summarise_submission_csv(df):
    text = ""
    df["Class"] = df["ImageId_ClassId"].str[-1].astype(np.int32)
    df["Label"] = (df["EncodedPixels"] != "").astype(np.int32)
    num_image = len(df) // 4
    num = len(df)

    pos = (df["Label"] == 1).sum()
    neg = num - pos

    pos1 = ((df["Class"] == 1) & (df["Label"] == 1)).sum()
    pos2 = ((df["Class"] == 2) & (df["Label"] == 1)).sum()
    pos3 = ((df["Class"] == 3) & (df["Label"] == 1)).sum()
    pos4 = ((df["Class"] == 4) & (df["Label"] == 1)).sum()

    neg1 = num_image - pos1
    neg2 = num_image - pos2
    neg3 = num_image - pos3
    neg4 = num_image - pos4

    text += "compare with LB probing ... \n"
    text += "\t\tnum_image = %5d(1801) \n" % num_image
    text += "\t\tnum  = %5d(7204) \n" % num
    text += "\n"

    text += "\t\tpos1 = %5d( 128)  %0.3f\n" % (pos1, pos1 / 128)
    text += "\t\tpos2 = %5d(  43)  %0.3f\n" % (pos2, pos2 / 43)
    text += "\t\tpos3 = %5d( 741)  %0.3f\n" % (pos3, pos3 / 741)
    text += "\t\tpos4 = %5d( 120)  %0.3f\n" % (pos4, pos4 / 120)
    text += "\n"

    text += "\t\tneg1 = %5d(1673)  %0.3f  %3d\n" % (neg1, neg1 / 1673, neg1 - 1673)
    text += "\t\tneg2 = %5d(1758)  %0.3f  %3d\n" % (neg2, neg2 / 1758, neg2 - 1758)
    text += "\t\tneg3 = %5d(1060)  %0.3f  %3d\n" % (neg3, neg3 / 1060, neg3 - 1060)
    text += "\t\tneg4 = %5d(1681)  %0.3f  %3d\n" % (neg4, neg4 / 1681, neg4 - 1681)
    text += "--------------------------------------------------\n"
    text += "\t\tneg  = %5d(6172)  %0.3f  %3d \n" % (neg, neg / 6172, neg - 6172)
    text += "\n"

    if 1:
        # compare with reference
        pass

    return text


#########################################################################3


def compute_metric_label(truth_label, predict_label):
    t = truth_label.reshape(-1, 4)
    p = predict_label.reshape(-1, 4)

    #           num_truth, num_predict, correct, recall, precision
    # (all) neg
    # (all) pos
    #      neg1
    #      pos1
    #     ...
    ts = np.array(
        [
            [1 - t.reshape(-1), t.reshape(-1)],
            [1 - t[:, 0], t[:, 0]],
            [1 - t[:, 1], t[:, 1]],
            [1 - t[:, 2], t[:, 2]],
            [1 - t[:, 3], t[:, 3]],
        ]
    ).reshape(-1)
    ps = np.array(
        [
            [1 - p.reshape(-1), p.reshape(-1)],
            [1 - p[:, 0], p[:, 0]],
            [1 - p[:, 1], p[:, 1]],
            [1 - p[:, 2], p[:, 2]],
            [1 - p[:, 3], p[:, 3]],
        ]
    ).reshape(-1)

    result = []
    for tt, pp in zip(ts, ps):
        num_truth = tt.sum()
        num_predict = pp.sum()
        num_correct = (tt * pp).sum()
        recall = num_correct / num_truth
        precision = num_correct / num_predict
        result.append([num_truth, num_predict, num_correct, recall, precision])

    ## from kaggle probing ...
    kaggle_pos = np.array([128, 43, 741, 120])
    kaggle_neg_all = 6172
    kaggle_all = 1801 * 4

    recall_neg_all = result[0][3]
    recall_pos = np.array([result[3][3], result[5][3], result[7][3], result[9][3]])

    kaggle = []
    for dice_pos in [1.00, 0.75, 0.50]:
        k = (
            recall_neg_all * kaggle_neg_all + sum(dice_pos * recall_pos * kaggle_pos)
        ) / kaggle_all
        kaggle.append([k, dice_pos])

    return kaggle, result


def summarise_metric_label(kaggle, result):
    text = ""
    text += "* image level metric *\n"
    text += "             num_truth, num_predict,  num_correct,   recall, precision\n"
    text += "       neg1      %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[2],
    )
    text += "       neg2      %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[4],
    )
    text += "       neg3      %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[6],
    )
    text += "       neg4      %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[8],
    )
    text += "-----------------------------------------------------------------------\n"
    text += " (all) neg       %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[0],
    )
    text += "\n"

    text += "       pos1      %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[3],
    )
    text += "       pos2      %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[5],
    )
    text += "       pos3      %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[7],
    )
    text += "       pos4      %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[9],
    )
    text += "-----------------------------------------------------------------------\n"
    text += " (all) pos       %4d        %4d           %4d       %0.3f    %0.3f  \n" % (
        *result[1],
    )
    text += "\n"

    text += "kaggle = %0.5f @ dice%0.3f\n" % (kaggle[0][0], kaggle[0][1])
    text += "       = %0.5f @ dice%0.3f\n" % (kaggle[1][0], kaggle[1][1])
    text += "       = %0.5f @ dice%0.3f\n" % (kaggle[2][0], kaggle[2][1])
    text += "\n"

    return text


# ---

# ---
def compute_eer(fpr, tpr, threshold):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1 - tpr
    abs_diff = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diff)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, threshold[min_index]


def compute_roc_label(truth_label, probability_label):
    t = truth_label.reshape(-1, 4)
    p = probability_label.reshape(-1, 4)

    auc = []
    result = []
    for c in [0, 1, 2, 3]:
        fpr, tpr, threshold = sklearn_metrics.roc_curve(t[:, c], p[:, c])
        result.append([fpr, tpr, threshold])

        eer, threshold_eer = compute_eer(fpr, tpr, threshold)
        a = sklearn_metrics.roc_auc_score(t[:, c], p[:, c])
        auc.append([a, eer, threshold_eer])

    return auc, result


def summarise_roc_label(auc, result):
    text = ""

    for c, (a, eer, threshold_eer) in enumerate(auc):
        text += "class%d  : auc = %0.5f,  eer = %0.5f (%3d)" % (
            (c + 1),
            a,
            eer,
            threshold_eer,
        )
        text += "\n"
    text += "\n"

    # text += 'auc = %s\n'%(str(auc))
    if 0:
        text += "\n"
        for c, (fpr, tpr, threshold) in enumerate(result):
            text += "class%d\n" % (c + 1)
            text += "bin\tfpr\ttpr\n"
            for f, t, b in zip(fpr, tpr, threshold):
                text += "%0.0f\t%0.3f\t%0.3f\n" % (b, f, t)
            text += "\n"
        text += "\n"

    return text


def compute_metric_mask(truth, predict):
    num = len(truth)
    t = truth.reshape(num * 4, -1)
    p = predict.reshape(num * 4, -1)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)

    pos_index = (t_sum > 0).reshape(num, 4)
    neg_index = ~pos_index

    ## pixel mask level metrics
    pt_sum = (p * t).sum(-1)

    d_neg = (t_sum + p_sum == 0).astype(np.float32)
    d_pos = 2 * pt_sum / (t_sum + p_sum + 1e-12)
    d_neg = d_neg.reshape(num, 4)
    d_pos = d_pos.reshape(num, 4)

    h_neg = (p_sum == 0).astype(np.float32)
    h_pos = (p_sum > 0).astype(np.float32)
    h_neg = h_neg.reshape(num, 4)
    h_pos = h_pos.reshape(num, 4)

    t_neg = (t_sum == 0).astype(np.float32)
    t_pos = (t_sum > 0).astype(np.float32)
    t_neg = t_neg.reshape(num, 4)
    t_pos = h_pos.reshape(num, 4)

    #       num			     dice (mask)	    hit (label)
    # all	neg (%)	pos (%)	 all	neg	pos	    all	neg	pos

    result = []
    for c in range(4):
        num_pos = pos_index[:, c].sum()
        num_neg = neg_index[:, c].sum()

        dice_neg = d_neg[:, c][neg_index[:, c]].sum()
        dice_pos = d_pos[:, c][pos_index[:, c]].sum()
        hit_neg = h_neg[:, c][neg_index[:, c]].sum()
        hit_pos = h_pos[:, c][pos_index[:, c]].sum()

        truth_neg = t_neg[:, c].sum()
        truth_pos = t_pos[:, c].sum()
        predict_neg = h_neg[:, c].sum()
        predict_pos = h_pos[:, c].sum()

        result.append(
            [
                num_neg,
                num_neg / num,
                num_pos,
                num_pos / num,
                (dice_neg + dice_pos) / num,
                dice_neg / num_neg,
                dice_pos / num_pos,
                (hit_neg + hit_pos) / num,
                hit_neg / num_neg,
                hit_pos / num_pos,
                hit_neg / predict_neg,
                hit_pos / predict_pos,
            ]
        )

    # ----

    ## from kaggle probing ...
    kaggle_pos = np.array([128, 43, 741, 120])
    kaggle_neg_all = 6172
    kaggle_all = 1801 * 4

    dice_neg_all = (
        result[0][5] * result[0][1]
        + result[1][5] * result[1][1]
        + result[2][5] * result[2][1]
        + result[3][5] * result[3][1]
    ) / (result[0][1] + result[1][1] + result[2][1] + result[3][1])
    dice_pos = np.array([result[0][6], result[1][6], result[2][6], result[3][6]])
    kaggle = (dice_neg_all * kaggle_neg_all + sum(dice_pos * kaggle_pos)) / kaggle_all
    return kaggle, result


def summarise_metric_mask(kaggle, result):
    text = ""
    text += "         num_neg,      pos      | dice_all  neg    pos  | hit_all  neg   pos    | prec_neg   pos   \n"
    text += "--------------------------------------------------------------------------------------------------------\n"
    text += (
        "class1  %4d(%0.2f)  %4d(%0.2f)  |  %0.2f   %0.2f   %0.2f   |  %0.2f   %0.2f   %0.2f   |  %0.3f   %0.3f\n"
        % (*result[0],)
    )
    text += (
        "class2  %4d(%0.2f)  %4d(%0.2f)  |  %0.2f   %0.2f   %0.2f   |  %0.2f   %0.2f   %0.2f   |  %0.3f   %0.3f\n"
        % (*result[1],)
    )
    text += (
        "class3  %4d(%0.2f)  %4d(%0.2f)  |  %0.2f   %0.2f   %0.2f   |  %0.2f   %0.2f   %0.2f   |  %0.3f   %0.3f\n"
        % (*result[2],)
    )
    text += (
        "class4  %4d(%0.2f)  %4d(%0.2f)  |  %0.2f   %0.2f   %0.2f   |  %0.2f   %0.2f   %0.2f   |  %0.3f   %0.3f\n"
        % (*result[3],)
    )

    text += "\n"
    text += "kaggle = %0.5f\n" % (kaggle)
    text += "\n"

    return text


## draw  ###########################################################################

# DEFECT_COLOR = [(0,0,0),(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
DEFECT_COLOR = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 128, 0), (0, 255, 255)]


def mask_to_inner_contour(mask):
    mask = mask > 0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), "reflect")
    contour = mask & (
        (pad[1:-1, 1:-1] != pad[:-2, 1:-1])
        | (pad[1:-1, 1:-1] != pad[2:, 1:-1])
        | (pad[1:-1, 1:-1] != pad[1:-1, :-2])
        | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
    )
    return contour


def draw_contour_overlay(image, mask, color=(0, 0, 255), thickness=1):
    contour = mask_to_inner_contour(mask)
    if thickness == 1:
        image[contour] = color
    else:
        for y, x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x, y), thickness // 2, color, lineType=cv2.LINE_4)
    return image


# https://en.wikipedia.org/wiki/Blend_modes
#
#
def draw_mask_overlay(image, mask, color=(0, 0, 255), alpha=0.5):
    H, W, C = image.shape
    mask = (mask * alpha).reshape(H, W, 1)
    overlay = image.astype(np.float32)
    overlay = np.maximum(overlay, mask * color)
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


def draw_grid(image, grid_size=[32, 32], color=[64, 64, 64], thickness=1):
    H, W, C = image.shape
    dx, dy = grid_size

    for x in range(0, W, dx):
        cv2.line(image, (x, 0), (x, H), color, thickness=thickness)
    for y in range(0, H, dy):
        cv2.line(image, (0, y), (W, y), color, thickness=thickness)
    return image


##----


def draw_truth_mask_overlay(image, truth_mask):
    H, W, C = image.shape
    for c in [1, 2, 3, 4]:
        t = truth_mask == c
        image = draw_contour_overlay(image, t, DEFECT_COLOR[c], thickness=2)
    return image


##----


def draw_predict_result_label(image, truth_label, truth_mask, probability_label):
    color = DEFECT_COLOR
    H, W, C = image.shape
    truth_label = truth_label.reshape(-1)
    truth_mask = truth_mask.reshape(H, W)
    probability_label = probability_label.reshape(-1)

    overlay = image.copy()

    for c in [1, 2, 3, 4]:
        t = truth_mask == c
        overlay = draw_contour_overlay(overlay, t, color[c], thickness=2)

    for c in range(4):
        asterisk = "*" if probability_label[c] > 0.5 else ""
        if (probability_label[c] > 0.5) * (truth_label[c] < 0.5):
            asterisk = "FP!"
        if (probability_label[c] < 0.5) * (truth_label[c] > 0.5):
            asterisk = "MISS!"
        draw_shadow_text(
            overlay,
            "pos%d %d %0.2f %s  "
            % (c + 1, truth_label[c], probability_label[c], asterisk),
            (5, (c + 1) * 30),
            1,
            color[c + 1],
            2,
        )

    # draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = np.vstack([image, overlay])
    result = draw_grid(result, grid_size=[W, H], color=[255, 255, 255], thickness=1)
    return result


def draw_predict_result_label_one_defect_only(
    image, truth_label, truth_mask, probability_label
):
    color = DEFECT_COLOR
    H, W, C = image.shape
    truth_label = truth_label.reshape(-1)
    truth_mask = truth_mask.reshape(H, W)
    probability_label = probability_label.reshape(-1)

    overlay = image.copy()

    for c in [1, 2, 3, 4]:
        t = truth_mask == c
        overlay = draw_contour_overlay(overlay, t, color[c], thickness=2)

    if 1:
        asterisk = "*" if probability_label > 0.5 else ""
        if (probability_label > 0.5) * (truth_label < 0.5):
            asterisk = "FP!"
        if (probability_label < 0.5) * (truth_label > 0.5):
            asterisk = "MISS!"
        draw_shadow_text(
            overlay,
            "pos%d %d %0.2f %s  " % (1, truth_label, probability_label, asterisk),
            (5, (1) * 30),
            1,
            color[1],
            2,
        )

    # draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = np.vstack([image, overlay])
    result = draw_grid(result, grid_size=[W, H], color=[255, 255, 255], thickness=1)
    return result


def draw_predict_result_attention(
    image, truth_label, truth_mask, truth_attention, probability_label, attention
):
    color = DEFECT_COLOR
    H, W, C = image.shape

    truth_label = truth_label.reshape(-1)
    truth_mask = truth_mask.reshape(H, W)
    probability_label = probability_label.reshape(-1)

    overlay = image.copy()
    result = []
    for c in range(4):
        r = np.zeros((H, W, 3), np.uint8)

        t = truth_mask == (c + 1)
        ta = truth_attention[c]
        pa = attention[c]
        ta = cv2.resize(ta, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        pa = cv2.resize(pa, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

        r = draw_mask_overlay(r, pa, (255, 255, 255), alpha=1)
        r = draw_contour_overlay(r, t, color[c + 1], thickness=4)

        overlay = draw_mask_overlay(overlay, ta, color[c + 1], alpha=0.75)
        overlay = draw_contour_overlay(overlay, t, color[c + 1], thickness=4)

        draw_shadow_text(r, "predict%d" % (c + 1), (5, 30), 1, color[c + 1], 2)
        # draw_shadow_text(r,'[%d] %0.03f'%(truth_label[c], probability_label[c]),(5,60),1,color[c+1],2)
        if truth_label[c] > 0.5:
            draw_shadow_text(r, "[%d] " % (truth_label[c]), (5, 60), 1, color[c + 1], 2)
        else:
            draw_shadow_text(r, "[%d] " % (truth_label[c]), (5, 60), 1, [64, 64, 64], 2)

        draw_shadow_text(
            r,
            "%0.03f" % (probability_label[c]),
            (60, 60),
            1,
            int_tuple([max(probability_label[c], 0.25) * 255] * 3),
            2,
        )

        result.append(r)

    # ---
    draw_shadow_text(overlay, "truth", (5, 30), 1, [255, 255, 255], 2)
    result = [image, overlay] + result
    result = np.vstack(result)
    result = draw_grid(result, grid_size=[W, H], color=[255, 255, 255], thickness=3)

    return result


def draw_predict_result_8cls(
    image, truth_label, truth_mask, truth_attention, probability_attention
):
    color = DEFECT_COLOR
    H, W, C = image.shape
    truth_label = truth_label.reshape(-1)
    truth_mask = truth_mask.reshape(H, W)
    truth_attention = truth_attention.reshape(4, -1)
    probability_attention = probability_attention.reshape(4, -1)

    length = truth_attention.shape[1]
    w = W // length

    overlay = image.copy()
    for c in [1, 2, 3, 4]:
        t = truth_mask == c
        overlay = draw_contour_overlay(overlay, t, color[c], thickness=2)

    overlay1 = np.zeros_like(overlay)
    for c in [1, 2, 3, 4]:
        for i in range(length):
            t = truth_attention[c - 1, i]
            p = probability_attention[c - 1, i]

            asterisk = "*" if p > 0.5 else ""
            if (p > 0.5) * (t < 0.5):
                asterisk = "FP!"
            if (p < 0.5) * (t > 0.5):
                asterisk = "MISS!"

            clr = [64, 64, 64]

            if (p > 0.5) or (t > 0.5):
                clr = color[c]
            if i == 0:
                draw_shadow_text(
                    overlay1,
                    "pos%d [%d] %0.2f %s  " % (c, t, p, asterisk),
                    (5 + i * w, (c) * 30),
                    1,
                    clr,
                    2,
                )
            else:
                draw_shadow_text(
                    overlay1,
                    "[%d] %0.2f %s  " % (t, p, asterisk),
                    (5 + i * w, (c) * 30),
                    1,
                    clr,
                    2,
                )

    overlay = draw_grid(overlay, grid_size=[w, H], color=[255, 255, 255], thickness=1)
    overlay1 = draw_grid(overlay1, grid_size=[w, H], color=[255, 255, 255], thickness=1)

    # draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = np.vstack([image, overlay, overlay1])
    result = draw_grid(result, grid_size=[W, H], color=[255, 255, 255], thickness=1)
    return result


def draw_predict_result_softmax_label(
    image, truth_label, truth_mask, probability_label
):
    color = DEFECT_COLOR
    H, W, C = image.shape
    truth_label = truth_label.reshape(-1)
    truth_mask = truth_mask.reshape(H, W)
    probability_label = probability_label.reshape(-1)

    overlay = image.copy()

    for c in [1, 2, 3, 4]:
        t = truth_mask == c
        overlay = draw_contour_overlay(overlay, t, color[c], thickness=2)

    for c in [0]:
        is_empty = int(sum(truth_label) == 0)
        draw_shadow_text(
            overlay,
            "neg  %d %0.2f  " % (is_empty, probability_label[0]),
            (5, 30),
            1,
            color[c],
            2,
            color1=[255, 255, 255],
            thickness1=3,
        )

    for c in [1, 2, 3, 4]:
        asterisk = "*" if probability_label[c] > 0.5 else ""
        if (probability_label[c] > 0.5) * (truth_label[c - 1] < 0.5):
            asterisk = "FP!"
        if (probability_label[c] < 0.5) * (truth_label[c - 1] > 0.5):
            asterisk = "MISS!"
        draw_shadow_text(
            overlay,
            "pos%d %d %0.2f %s  "
            % (c, truth_label[c - 1], probability_label[c], asterisk),
            (5, (c + 1) * 30),
            1,
            color[c],
            2,
        )

    # draw_shadow_text(overlay,'truth',(5,30),1,[255,255,255],2)
    result = np.vstack([image, overlay])
    result = draw_grid(result, grid_size=[W, H], color=[255, 255, 255], thickness=1)
    return result


def draw_predict_result(
    image, truth_label, truth_mask, probability_label, probability_mask
):
    color = DEFECT_COLOR
    H, W, C = image.shape
    truth_label = truth_label.reshape(-1)
    truth_mask = truth_mask.reshape(H, W)
    probability_label = probability_label.reshape(-1)
    probability_mask = probability_mask.reshape(5, H, W)

    overlay = image.copy()
    result = []

    for c in [1, 2, 3, 4]:
        t = truth_mask == c
        p = probability_mask[c]
        l = truth_label[c - 1]
        u = probability_label[c - 1]

        # ---
        overlay = draw_contour_overlay(overlay, t, color[c], thickness=2)

        # ---
        r = np.zeros((H, W, 3), np.uint8)
        r = draw_mask_overlay(r, p, (255, 255, 255), alpha=1)
        r = draw_contour_overlay(r, t, color[c], thickness=2)
        # draw_shadow_text(r,'predict%d'%(c),(5,30),1,color[c],2)

        asterisk = ""
        if (l > 0.5) * (u > 0.5):
            asterisk = "*"
        if (l > 0.5) * (u < 0.5):
            asterisk = "MISS!"
        if (l < 0.5) * (u > 0.5):
            asterisk = "FP!"
        draw_shadow_text(r, "predict%d" % (c), (5, 30), 1, color[c], 2)
        draw_shadow_text(r, "[%d] " % (l), (5, 60), 1, color[c], 2)
        draw_shadow_text(
            r,
            "%0.03f %s" % (u, asterisk),
            (60, 60),
            1,
            int_tuple([max(u, 0.25) * 255] * 3),
            2,
        )
        # ---
        result.append(r)

    draw_shadow_text(overlay, "truth", (5, 30), 1, [255, 255, 255], 2)
    result = [image, overlay, *result]
    result = np.vstack(result)
    result = draw_grid(result, grid_size=[W, H], color=[255, 255, 255], thickness=1)
    return result


## check function  ############################################################################################


def run_check_rle():
    # https://www.kaggle.com/bigkizd/se-resnext50-89
    def ref_mask2rle(img):
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return " ".join(str(x) for x in runs)

    image = cv2.imread(
        "/root/share/project/kaggle/2019/steel/data/train_images/002fc4e19.jpg",
        cv2.IMREAD_COLOR,
    )
    value = [
        "002fc4e19.jpg_1",
        "146021 3 146275 10 146529 40 146783 46 147038 52 147292 59 147546 65 147800 70 148055 71 148311 72 148566 73 148822 74 149077 75 149333 76 149588 77 149844 78 150100 78 150357 75 150614 72 150870 70 151127 67 151384 64 151641 59 151897 53 152154 46 152411 22",
        "002fc4e19.jpg_2",
        "145658 7 145901 20 146144 33 146386 47 146629 60 146872 73 147115 86 147364 93 147620 93 147876 93 148132 93 148388 93 148644 93 148900 93 149156 93 149412 93 149668 46",
        "002fc4e19.jpg_3",
        "",
        "002fc4e19.jpg_4",
        "",
    ]
    rle = [value[i] for i in range(1, 8, 2)]

    mask = np.array(
        [run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle]
    )
    print(mask.shape)

    print("**run_length_encode**")
    rle1 = [run_length_encode(m) for m in mask]
    print("0", rle1[0])
    print("1", rle1[1])
    print("2", rle1[2])
    print("3", rle1[3])
    assert rle1 == rle
    print("check ok!!!!")

    print("**ref_mask2rle**")
    rle2 = [ref_mask2rle(m) for m in mask]
    print("0", rle2[0])
    print("1", rle2[1])
    print("2", rle2[2])
    print("3", rle2[3])
    assert rle2 == rle
    print("check ok!!!!")

    exit(0)

    image_show_norm("mask[0]", mask[0], 0, 1)
    image_show_norm("mask[1]", mask[1], 0, 1)
    image_show_norm("mask[2]", mask[2], 0, 1)
    image_show_norm("mask[3]", mask[3], 0, 1)
    image_show("image", image)

    # ---
    mask0 = draw_mask_overlay(image, mask[0], color=(0, 0, 255))
    image_show("mask0", mask0)
    mask1 = draw_mask_overlay(image, mask[1], color=(0, 0, 255))
    image_show("mask1", mask1)

    cv2.waitKey(0)


def run_show_csv():
    df = pd.read_csv(
        "/root/share/project/kaggle/2019/steel/data/semi/dump/merge-x11-14-fix-0.91262.csv"
    ).fillna("")
    text = summarise_submission_csv(df)
    print("%s" % (text))


"""

/root/share/project/kaggle/2019/steel/data/semi/dump/merge-x11-14-fix-0.91262.csv
compare with LB probing ... 
		num_image =  1801(1801) 
		num  =  7204(7204) 
		neg  =  6375(6172)  0.885 
		pos  =   829(1032)  0.115 
		pos1 =   112( 128)  0.062  0.135 
		pos2 =    23(  43)  0.013  0.028 
		pos3 =   585( 741)  0.325  0.706 
		pos4 =   109( 120)  0.061  0.131 

"""
# main #################################################################
if __name__ == "__main__":
    print("%s: calling main function ... " % os.path.basename(__file__))

    run_check_rle()
    # run_make_test_split1()
    # run_make_dummy()

    # run_make_train_split()
