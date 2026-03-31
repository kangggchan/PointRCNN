import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _init_path
import os
import argparse
import pickle

import numpy as np
import torch

import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.datasets.kitti_dataset import KittiDataset
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from tools.preprocess_dataset import preprocess_dataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value

    candidate_paths = [
        os.path.abspath(path_value),
        os.path.abspath(os.path.join(REPO_ROOT, path_value)),
        os.path.abspath(os.path.join(REPO_ROOT, '..', path_value)),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return candidate_paths[0]


def _class_list_from_arg(class_name):
    return KittiRCNNDataset.parse_classes(class_name)


def _class_tag(class_list):
    return class_list[0] if len(class_list) == 1 else 'multi'


class GTDatabaseGenerator(KittiDataset):
    def __init__(self, root_dir, split='train', classes='Car'):
        super().__init__(root_dir, split=split)
        class_list = _class_list_from_arg(classes)
        self.classes = tuple(['Background'] + class_list)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def filtrate_objects(self, obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.classes:
                continue
            if obj.cls_type.lower() in ['box', 'boxes', 'bbox']:
                continue
            if obj.level_str not in ['Easy', 'Moderate', 'Hard']:
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    def generate(self, save_file_name):
        gt_database = []
        for sample_id_str in self.image_idx_list:
            sample_id = int(sample_id_str)
            print(f'process gt sample (id={sample_id:06d})')

            pts_lidar = self.get_lidar(sample_id)
            calib = self.get_calib(sample_id)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3]

            obj_list = self.filtrate_objects(self.get_label(sample_id))
            gt_boxes3d = np.zeros((len(obj_list), 7), dtype=np.float32)
            for k, obj in enumerate(obj_list):
                gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] = \
                    obj.pos, obj.h, obj.w, obj.l, obj.ry

            if len(gt_boxes3d) == 0:
                continue

            boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(
                torch.from_numpy(pts_rect), torch.from_numpy(gt_boxes3d)
            )

            for k in range(len(boxes_pts_mask_list)):
                pt_mask_flag = (boxes_pts_mask_list[k].numpy() == 1)
                obj_pts = pts_rect[pt_mask_flag]
                if obj_pts.shape[0] == 0:
                    print(f'  Skipping empty box: {obj_list[k].cls_type} in sample {sample_id:06d}')
                    continue
                gt_database.append({
                    'sample_id': sample_id,
                    'cls_type': obj_list[k].cls_type,
                    'gt_box3d': gt_boxes3d[k],
                    'points': obj_pts.astype(np.float32),
                    'intensity': pts_intensity[pt_mask_flag].astype(np.float32),
                    'obj': obj_list[k],
                })

        with open(save_file_name, 'wb') as f:
            pickle.dump(gt_database, f)

        print(f'Save gt_database ({len(gt_database)} objects) to {save_file_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./gt_database')
    parser.add_argument('--class_name', type=str, default='Car,Human,ForkLift,CargoBike,ELFplusplus,FTS')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--root_dir', type=str, default='../data/dataset')
    parser.add_argument('--skip_preprocess', action='store_true', default=False)
    args = parser.parse_args()

    class_list = _class_list_from_arg(args.class_name)
    root_dir_abs = _resolve_path(args.root_dir)
    split_file = os.path.join(root_dir_abs, 'KITTI', 'ImageSets', f'{args.split}.txt')
    if not args.skip_preprocess and not os.path.exists(split_file):
        preprocess_dataset(root_dir_abs, classes=tuple(class_list))

    args.save_dir = _resolve_path(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    save_file_name = os.path.join(args.save_dir, f'{args.split}_gt_database_3level_{_class_tag(class_list)}.pkl')

    dataset = GTDatabaseGenerator(root_dir=root_dir_abs, split=args.split, classes=args.class_name)
    dataset.generate(save_file_name)


if __name__ == '__main__':
    main()
