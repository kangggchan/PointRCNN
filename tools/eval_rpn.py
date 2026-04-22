import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _init_path
import argparse
import glob
import logging
import re
import time
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import tqdm

from lib.config import cfg, cfg_from_file, cfg_from_list, save_config_to_file
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.net.point_rcnn import PointRCNN
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils
import tools.train_utils.train_utils as train_utils


def _torch_load_compat(filename):
    try:
        return torch.load(filename, weights_only=False)
    except TypeError:
        return torch.load(filename)


parser = argparse.ArgumentParser(description='Evaluate RPN proposals and optionally export offline RCNN features.')
parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='Config file for evaluation.')
parser.add_argument('--eval_all', action='store_true', default=False, help='Evaluate every checkpoint in ckpt_dir.')
parser.add_argument('--test', action='store_true', default=False, help='Evaluate without ground truth labels.')
parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to evaluate.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation.')
parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers.')
parser.add_argument('--data_root', type=str, default='../data/dataset',
                    help='Dataset root containing KITTI/ for evaluation.')
parser.add_argument('--extra_tag', type=str, default='default', help='Optional tag appended to the output path.')
parser.add_argument('--output_dir', type=str, default=None, help='Override output directory.')
parser.add_argument('--ckpt_dir', type=str, default=None, help='Directory used with --eval_all.')
parser.add_argument('--save_result', action='store_true', default=False, help='Save detections and segmentation results.')
parser.add_argument('--save_rpn_feature', action='store_true', default=False,
                    help='Save features and proposals for offline RCNN training or validation.')
parser.add_argument('--random_select', action='store_true', default=True, help='Sample to the same number of points.')
parser.add_argument('--start_epoch', default=0, type=int, help='Ignore checkpoints smaller than this epoch.')
parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='Extra config overrides, for example TEST.SPLIT train_aug')
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def _get_gt_boxes_and_classes(dataset, sample_id, gt_boxes3d):
    if isinstance(gt_boxes3d, torch.Tensor):
        gt_boxes_np = gt_boxes3d.detach().cpu().numpy()
    else:
        gt_boxes_np = np.asarray(gt_boxes3d)

    if gt_boxes_np.size == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    gt_boxes_np = gt_boxes_np.reshape(gt_boxes_np.shape[0], -1)
    gt_boxes = gt_boxes_np[:, 0:7].astype(np.float32)
    valid_mask = np.abs(gt_boxes).sum(axis=1) > 0
    gt_boxes = gt_boxes[valid_mask]

    if gt_boxes_np.shape[1] > 7:
        gt_class_ids = gt_boxes_np[:, 7].astype(np.int64)[valid_mask]
        return gt_boxes, gt_class_ids

    filtered_labels = dataset.filtrate_objects(dataset.get_label(sample_id))
    gt_class_ids = np.array([dataset.get_class_idx(obj.cls_type) for obj in filtered_labels], dtype=np.int64)
    if gt_class_ids.shape[0] != gt_boxes.shape[0]:
        gt_class_ids = gt_class_ids[:gt_boxes.shape[0]]
    return gt_boxes, gt_class_ids


def _class_names_from_ids(class_ids, class_names):
    det_names = []
    for class_id in class_ids:
        class_idx = int(class_id)
        if 0 <= class_idx < len(class_names):
            det_names.append(class_names[class_idx])
        else:
            det_names.append(class_names[0])
    return det_names


def save_kitti_format(sample_id, calib, bbox3d, output_dir, scores, img_shape, det_class_names=None,
                      default_class_name='Car'):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)
    box_valid_mask &= np.all(np.isfinite(bbox3d[:, 3:6]), axis=1)
    box_valid_mask &= np.all(bbox3d[:, 3:6] > 0, axis=1)

    output_file = os.path.join(output_dir, '%06d.txt' % sample_id)
    with open(output_file, 'w') as handle:
        for idx in range(bbox3d.shape[0]):
            if box_valid_mask[idx] == 0:
                continue

            x, z, ry = bbox3d[idx, 0], bbox3d[idx, 2], bbox3d[idx, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry
            class_name = det_class_names[idx] if det_class_names is not None else default_class_name

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' % (
                class_name,
                alpha,
                img_boxes[idx, 0],
                img_boxes[idx, 1],
                img_boxes[idx, 2],
                img_boxes[idx, 3],
                bbox3d[idx, 3],
                bbox3d[idx, 4],
                bbox3d[idx, 5],
                bbox3d[idx, 0],
                bbox3d[idx, 1],
                bbox3d[idx, 2],
                bbox3d[idx, 6],
                scores[idx],
            ), file=handle)


def save_rpn_features(seg_result, rpn_scores_raw, pts_features, backbone_xyz, backbone_features, output_dir, sample_id):
    pts_intensity = pts_features[:, 0]

    np.save(os.path.join(output_dir, '%06d.npy' % sample_id), backbone_features)
    np.save(os.path.join(output_dir, '%06d_xyz.npy' % sample_id), backbone_xyz)
    np.save(os.path.join(output_dir, '%06d_seg.npy' % sample_id), seg_result)
    np.save(os.path.join(output_dir, '%06d_intensity.npy' % sample_id), pts_intensity)
    np.save(os.path.join(output_dir, '%06d_rawscore.npy' % sample_id), rpn_scores_raw)


def eval_one_epoch(model, dataloader, epoch_id, result_dir, logger):
    np.random.seed(1024)
    mode = 'TEST' if args.test else 'EVAL'

    if args.save_rpn_feature:
        feature_dir = os.path.join(result_dir, 'features')
        os.makedirs(feature_dir, exist_ok=True)

    if args.save_result or args.save_rpn_feature:
        det_dir = os.path.join(result_dir, 'detections', 'data')
        seg_dir = os.path.join(result_dir, 'seg_result')
        os.makedirs(det_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

    logger.info('---- EPOCH %s RPN EVALUATION ----' % epoch_id)
    model.eval()

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0

    dataset = dataloader.dataset
    num_classes = dataset.num_class - 1
    class_names = dataset.classes[1:]
    per_class_recalled_bbox_list = [[0] * 5 for _ in range(num_classes)]
    per_class_gt_bbox = [0] * num_classes

    cnt = 0
    rpn_iou_avg = 0.0
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval')

    for data in dataloader:
        sample_id_list = data['sample_id']
        pts_rect = data['pts_rect']
        pts_features = data['pts_features']
        pts_input = data['pts_input']
        cnt += len(sample_id_list)

        if not args.test:
            rpn_cls_label = torch.from_numpy(data['rpn_cls_label']).cuda(non_blocking=True).long()
            gt_boxes3d = data['gt_boxes3d']
            if gt_boxes3d.shape[1] > 0:
                gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking=True).float()

        inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        ret_dict = model({'pts_input': inputs})
        rpn_cls = ret_dict['rpn_cls']
        rpn_reg = ret_dict['rpn_reg']
        backbone_xyz = ret_dict['backbone_xyz']
        backbone_features = ret_dict['backbone_features']

        rpn_scores_raw, rpn_scores, _, pred_classes, _ = model.rpn.proposal_layer.get_point_cls_info(rpn_cls)
        if pred_classes is None:
            seg_result = (rpn_scores > cfg.RPN.SCORE_THRESH).long()
        else:
            seg_result = ((pred_classes > 0) & (rpn_scores > cfg.RPN.SCORE_THRESH)).long()

        rois, roi_scores_raw, roi_class_ids = model.rpn.proposal_layer(
            rpn_cls,
            rpn_reg,
            backbone_xyz,
            return_class_ids=True,
        )
        batch_size = rois.shape[0]

        for batch_idx in range(batch_size):
            cur_sample_id = sample_id_list[batch_idx]
            cur_scores_raw = roi_scores_raw[batch_idx]
            cur_boxes3d = rois[batch_idx]
            cur_roi_class_ids = roi_class_ids[batch_idx]
            cur_seg_result = seg_result[batch_idx]
            cur_pts_rect = pts_rect[batch_idx]

            if not args.test:
                cur_rpn_cls_label = rpn_cls_label[batch_idx]
                cur_gt_boxes3d = gt_boxes3d[batch_idx]

                gt_tail = cur_gt_boxes3d.__len__() - 1
                while gt_tail > 0 and cur_gt_boxes3d[gt_tail].sum() == 0:
                    gt_tail -= 1
                cur_gt_boxes3d = cur_gt_boxes3d[:gt_tail + 1]

                if cur_gt_boxes3d.shape[0] > 0:
                    cur_gt_boxes_np, gt_class_ids = _get_gt_boxes_and_classes(dataset, cur_sample_id, cur_gt_boxes3d)
                    if cur_gt_boxes_np.shape[0] > 0:
                        gt_class_ids = gt_class_ids - 1
                        iou3d = iou3d_utils.boxes_iou3d_gpu(
                            cur_boxes3d,
                            torch.from_numpy(cur_gt_boxes_np).cuda(non_blocking=True).float(),
                        )
                        gt_max_iou, _ = iou3d.max(dim=0)

                        for thresh_idx, thresh in enumerate(thresh_list):
                            total_recalled_bbox_list[thresh_idx] += (gt_max_iou > thresh).sum().item()
                            for class_idx in range(num_classes):
                                class_mask = gt_class_ids == class_idx
                                if class_mask.sum() > 0:
                                    per_class_recalled_bbox_list[class_idx][thresh_idx] += (
                                        gt_max_iou[class_mask] > thresh
                                    ).sum().item()

                        total_gt_bbox += cur_gt_boxes_np.shape[0]
                        for class_idx in range(num_classes):
                            per_class_gt_bbox[class_idx] += int((gt_class_ids == class_idx).sum())

                fg_mask = cur_rpn_cls_label > 0
                pred_fg_mask = cur_seg_result > 0
                correct = (pred_fg_mask & fg_mask).sum().float()
                union = fg_mask.sum().float() + pred_fg_mask.sum().float() - correct
                rpn_iou_avg += (correct / torch.clamp(union, min=1.0)).item()

            if args.save_rpn_feature:
                save_rpn_features(
                    seg_result[batch_idx].float().cpu().numpy(),
                    rpn_scores_raw[batch_idx].float().cpu().numpy(),
                    pts_features[batch_idx],
                    backbone_xyz[batch_idx].cpu().numpy(),
                    backbone_features[batch_idx].cpu().numpy().transpose(1, 0),
                    feature_dir,
                    cur_sample_id,
                )

            if args.save_result or args.save_rpn_feature:
                cur_pred_cls = pred_classes[batch_idx].cpu().numpy() if pred_classes is not None else cur_seg_result.cpu().numpy()
                output_file = os.path.join(seg_dir, '%06d.npy' % cur_sample_id)

                if not args.test:
                    cur_gt_cls = cur_rpn_cls_label.cpu().numpy()
                    output_data = np.concatenate(
                        (cur_pts_rect.reshape(-1, 3), cur_gt_cls.reshape(-1, 1), cur_pred_cls.reshape(-1, 1)), axis=1
                    )
                else:
                    output_data = np.concatenate((cur_pts_rect.reshape(-1, 3), cur_pred_cls.reshape(-1, 1)), axis=1)

                np.save(output_file, output_data.astype(np.float16))

                calib = dataset.get_calib(cur_sample_id)
                cur_boxes3d_np = cur_boxes3d.cpu().numpy()
                cur_scores_np = cur_scores_raw.cpu().numpy()
                cur_det_class_names = _class_names_from_ids(cur_roi_class_ids.cpu().numpy(), dataset.classes)
                image_shape = dataset.get_image_shape(cur_sample_id)
                save_kitti_format(
                    cur_sample_id,
                    calib,
                    cur_boxes3d_np,
                    det_dir,
                    cur_scores_np,
                    image_shape,
                    det_class_names=cur_det_class_names,
                    default_class_name=class_names[0] if class_names else 'Car',
                )

        progress_bar.set_postfix({
            'mode': mode,
            'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox),
            'rpn_iou': rpn_iou_avg / max(cnt, 1.0),
        })
        progress_bar.update()

    progress_bar.close()

    logger.info(str(datetime.now()))
    logger.info('-------------------performance of epoch %s---------------------' % epoch_id)
    logger.info('rpn iou avg: %f' % (rpn_iou_avg / max(cnt, 1.0)))

    ret_dict = {'rpn_iou': rpn_iou_avg / max(cnt, 1.0)}
    for thresh_idx, thresh in enumerate(thresh_list):
        cur_recall = total_recalled_bbox_list[thresh_idx] / max(total_gt_bbox, 1.0)
        logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' % (
            thresh,
            total_recalled_bbox_list[thresh_idx],
            total_gt_bbox,
            cur_recall,
        ))
        ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_recall

    logger.info('-------------------per-class recall---------------------')
    per_class_recall_file = os.path.join(result_dir, 'per_class_recall_epoch_%s.csv' % epoch_id)
    with open(per_class_recall_file, 'w') as handle:
        handle.write('class_name,gt_count,' + ','.join(['recall@%.1f' % thresh for thresh in thresh_list]) + '\n')
        for class_idx in range(num_classes):
            class_name = class_names[class_idx]
            logger.info('\n%s (total GT: %d):' % (class_name, per_class_gt_bbox[class_idx]))

            csv_line = '%s,%d' % (class_name, per_class_gt_bbox[class_idx])
            for thresh_idx, thresh in enumerate(thresh_list):
                if per_class_gt_bbox[class_idx] > 0:
                    class_recall = per_class_recalled_bbox_list[class_idx][thresh_idx] / per_class_gt_bbox[class_idx]
                else:
                    class_recall = 0.0
                logger.info('  recall(thresh=%.3f): %d / %d = %.4f' % (
                    thresh,
                    per_class_recalled_bbox_list[class_idx][thresh_idx],
                    per_class_gt_bbox[class_idx],
                    class_recall,
                ))
                csv_line += ',%.4f' % class_recall
                ret_dict['rpn_recall_%s(thresh=%.2f)' % (class_name, thresh)] = class_recall
            handle.write(csv_line + '\n')

    logger.info('\nPer-class recall saved to: %s' % per_class_recall_file)
    logger.info('result is saved to: %s' % result_dir)
    return ret_dict


def load_part_ckpt(model, filename, logger, total_keys=-1, strict_shape=False):
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    logger.info("==> Loading part model from checkpoint '%s'", filename)
    checkpoint = _torch_load_compat(filename)
    model_state = checkpoint.get('model_state', checkpoint)

    state_dict = model.state_dict()
    update_model_state = {}
    shape_mismatch_keys = []
    for key, val in model_state.items():
        if key not in state_dict:
            continue
        if state_dict[key].shape != val.shape:
            shape_mismatch_keys.append(key)
            continue
        update_model_state[key] = val

    if shape_mismatch_keys:
        preview = ', '.join(shape_mismatch_keys[:5])
        suffix = '...' if len(shape_mismatch_keys) > 5 else ''
        message = 'Found %d keys with shape mismatch: %s%s' % (len(shape_mismatch_keys), preview, suffix)
        if strict_shape:
            raise RuntimeError(message)
        logger.warning('==> Skipped %s', message)

    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)

    update_keys = len(update_model_state)
    if update_keys == 0:
        raise RuntimeError('No compatible keys found when loading partial checkpoint: %s' % filename)
    logger.info('==> Done (loaded %d/%d)' % (update_keys, total_keys))


def load_checkpoint(model, logger):
    if args.ckpt is None:
        return

    try:
        train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)
    except RuntimeError as err:
        msg = str(err)
        if 'Missing key(s)' in msg or 'size mismatch' in msg:
            logger.warning('Full checkpoint load failed, trying partial load for --ckpt')
            load_part_ckpt(
                model,
                filename=args.ckpt,
                logger=logger,
                total_keys=len(model.state_dict()),
                strict_shape=False,
            )
        else:
            raise


def eval_single_ckpt(root_result_dir):
    root_result_dir = os.path.join(root_result_dir, 'eval')
    epoch_src = args.ckpt
    num_list = re.findall(r'\d+', epoch_src) if epoch_src is not None else []
    epoch_id = num_list[-1] if num_list else 'no_number'
    root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % epoch_id, cfg.TEST.SPLIT)
    if args.test:
        root_result_dir = os.path.join(root_result_dir, 'test_mode')
    if args.extra_tag != 'default':
        root_result_dir = os.path.join(root_result_dir, args.extra_tag)
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    save_config_to_file(cfg, logger=logger)

    test_loader = create_dataloader(logger)
    model = PointRCNN(num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')
    model.cuda()

    backup_dir = os.path.join(root_result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp *.py %s/' % backup_dir)
    os.system('cp ../lib/net/*.py %s/' % backup_dir)
    os.system('cp ../lib/datasets/kitti_rcnn_dataset.py %s/' % backup_dir)

    load_checkpoint(model, logger)
    eval_one_epoch(model, test_loader, epoch_id, root_result_dir, logger)


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(item.strip()) for item in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if not num_list:
            continue

        epoch_id = num_list[-1]
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(root_result_dir, ckpt_dir):
    root_result_dir = os.path.join(root_result_dir, 'eval', 'eval_all_' + args.extra_tag)
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_eval_all_%s.txt' % cfg.TEST.SPLIT)
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    save_config_to_file(cfg, logger=logger)

    test_loader = create_dataloader(logger)
    model = PointRCNN(num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')
    model.cuda()

    backup_dir = os.path.join(root_result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp *.py %s/' % backup_dir)
    os.system('cp ../lib/net/*.py %s/' % backup_dir)
    os.system('cp ../lib/datasets/kitti_rcnn_dataset.py %s/' % backup_dir)

    ckpt_record_file = os.path.join(root_result_dir, 'eval_list_%s.txt' % cfg.TEST.SPLIT)
    with open(ckpt_record_file, 'a'):
        pass

    tb_log = SummaryWriter(log_dir=os.path.join(root_result_dir, 'tensorboard_%s' % cfg.TEST.SPLIT))

    while True:
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            print('Wait %s second for next check: %s' % (wait_second, ckpt_dir))
            time.sleep(wait_second)
            continue

        train_utils.load_checkpoint(model, filename=cur_ckpt)
        cur_result_dir = os.path.join(root_result_dir, 'epoch_%s' % cur_epoch_id, cfg.TEST.SPLIT)
        tb_dict = eval_one_epoch(model, test_loader, cur_epoch_id, cur_result_dir, logger)

        step = int(float(cur_epoch_id))
        if step == float(cur_epoch_id):
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, step)

        with open(ckpt_record_file, 'a') as handle:
            print('%s' % cur_epoch_id, file=handle)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def create_dataloader(logger):
    mode = 'TEST' if args.test else 'EVAL'
    test_set = KittiRCNNDataset(
        root_dir=args.data_root,
        npoints=cfg.RPN.NUM_POINTS,
        split=cfg.TEST.SPLIT,
        mode=mode,
        random_select=args.random_select,
        classes=cfg.CLASSES,
        logger=logger,
    )
    return DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=test_set.collate_batch,
    )


if __name__ == '__main__':
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]
    cfg.RPN.ENABLED = True
    cfg.RCNN.ENABLED = False

    root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
    ckpt_dir = os.path.join('../', 'output', 'rpn', cfg.TAG, 'ckpt')

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir
    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)

    with torch.no_grad():
        if args.eval_all:
            assert os.path.exists(ckpt_dir), '%s' % ckpt_dir
            repeat_eval_ckpt(root_result_dir, ckpt_dir)
        else:
            eval_single_ckpt(root_result_dir)
