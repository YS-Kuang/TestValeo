import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval

@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds

def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['vehicle', 'pedestrian', 'bicycle']
    ignored_gt, ignored_dt = [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i].lower()
        valid_class = -1

        if (gt_name == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        ignore = False
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)
        
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1

        if valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
        
    return num_valid_gt, ignored_gt, ignored_dt

def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_datas_list = []
    dt_datas_list = []
    ignored_gts, ignored_dets = [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det= rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        total_num_valid_gt += num_valid_gt
        gt_datas_num = len(gt_annos[i]["name"])
        gt_datas_list.append(gt_datas_num)

        dt_datas_score = dt_annos[i]["score"][..., np.newaxis]
        dt_datas_list.append(dt_datas_score)

    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, total_num_valid_gt)

# @numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 2], qboxes[j, 2]) - max(
                    boxes[i, 2] - boxes[i, 5], qboxes[j, 2] - qboxes[j, 5]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0
                    

def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]],
                               qboxes[:, [0, 1, 3, 4, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size 
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue

        det_idx = -1            
        valid_detection = NO_DETECTION      
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue


            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            
            assigned_detection[det_idx] = True
    
    
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]

def compute_statistics_jit1(overlaps,
                            gt_datas,
                            dt_datas,
                            ignored_gt,
                            ignored_det,
                            metric,
                            min_overlap,
                            thresh=0,
                            compute_fp=False,
                            compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas
    dt_scores = dt_datas

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size 
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue

        det_idx = -1            
        valid_detection = NO_DETECTION      
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue


            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            
            assigned_detection[det_idx] = True
    
    
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]

# @numba.jit(nopython=False)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             gt_datas,
                             dt_datas,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[i]
            dt_data = dt_datas[i]
            ignored_gt = ignored_gts[i]
            ignored_det = ignored_dets[i]
            tp, fp, fn, similarity, _ = compute_statistics_jit1(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]

def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]
    
def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=5):

    assert len(gt_annos) == len(dt_annos)

    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        
        if metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, -(np.pi / 2 + rots[..., np.newaxis])], axis = 1)
            
            
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)

        else:
            raise ValueError("unknown metric")
        
        parted_overlaps.append(overlap_part)
        example_idx += num_part

    overlaps = []
    example_idx = 0

    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
    
    
    return overlaps, parted_overlaps, total_gt_num, total_dt_num

def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=5):

    assert len(gt_annos) == len(dt_annos)

    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    N_SAMPLE_PTS = 41

    num_minoverlap = len(min_overlaps)            #得到长度为2
    num_class = len(current_classes)
    num_difficulty = len(difficultys)

    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, total_num_valid_gt) = rets
            
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                
                thresholdss = []
                
                for i in range(len(gt_annos)):
                    
                    rets = compute_statistics_jit(
                        overlaps[i],     # 单个图像的iou值b/n gt和dt
                        gt_datas_list[i],       # 是一个数，表示当前帧中的物体个数
                        dt_datas_list[i],       # N x 1阵列，表示的是预测得到的N个物体的得分情况
                        ignored_gts[i],         # 长度N数组，-1、0
                        ignored_dets[i],        # 长度N数组，-1、0
                        metric,                             # 0, 1, 或 2 (bbox, bev, 3d)
                        min_overlap=min_overlap,         # 浮动最小IOU阈值为正
                        thresh=0.0,                 # 忽略得分低于此值的dt。
                        compute_fp=False)

                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()
            

                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                
                pr = np.zeros([len(thresholds), 4])

                idx = 0
                for j,num_part in enumerate(split_parts):

                    gt_datas_part = np.array(gt_datas_list[idx:idx+num_part])
                    dt_datas_part = np.array(dt_datas_list[idx:idx+num_part])
                    ignored_dets_part = np.array(ignored_dets[idx:idx+num_part])
                    ignored_gts_part = np.array(ignored_gts[idx:idx+num_part])

                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx+num_part],
                        total_dt_num[idx:idx+num_part],
                        gt_datas_part,
                        dt_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds
                    )
                    idx += num_part

                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])

                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,             # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]  
        "precision": precision,          # RECALLING RECALL的顺序，因此精度降低
        "orientation": aos,
    }
    return ret_dict

def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100

def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100

def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()

def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None):

    # min_overlaps: [num_minoverlap, metric, num_class]
    #     #由上面得到的min_overlaps的形状是（2,3,5），
    # 这个是每个类别的IOU达到这个阈值时判断是否预测正确

    difficultys = [0, 1, 2]

    #metric: eval type. 0: bbox, 1: bev, 2: 3d
    # 重点是eval_class这个函数！！！！！！！！！！

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])

    return  mAP_3d, mAP_3d_R40

def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    
    overlap_0_7 = np.array([[0.7, 0.5, 0.5], [0.7, 0.5, 0.5],[0.7, 0.5, 0.5]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5], [0.5, 0.25, 0.25],[0.5, 0.25, 0.25]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 3]
    class_to_name = {
        0: 'Vehicle',
        1: 'Pedestrian',
        2: 'Bicycle'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    
    mAP_3d, mAP_3d_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)
    
    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            
            result += print_str((f"3d   AP:{mAP_3d[j, 0, i]:.4f}, "
                                 f"{mAP_3d[j, 1, i]:.4f}, "
                                 f"{mAP_3d[j, 2, i]:.4f}"))

            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))

            result += print_str((f"3d   AP:{mAP_3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP_3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP_3d_R40[j, 2, i]:.4f}"))

            if i == 0:
                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP_3d_R40[j, 2, 0]

    return result, ret_dict

