import os, sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _init_path
import os
import argparse
import pickle

import numpy as np
import torch

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from lib.datasets.kitti_dataset import KittiDataset
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.config import cfg
from tools.preprocess_dataset import preprocess_dataset

np.random.seed(1024)


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


def _parse_classes(class_name):
    return KittiRCNNDataset.parse_classes(class_name)


def _parse_class_weights(weight_str, class_names):
    """
    Parse class weights from string format.
    
    Args:
        weight_str: String in format "Class1:weight1,Class2:weight2,..." or "weight1,weight2,..." (positional)
        class_names: List of class names to map weights to
    
    Returns:
        Dict mapping class names to weights, or None if parsing fails
    """
    if weight_str is None:
        return None
    
    class_weights = {}
    pairs = weight_str.split(',')
    
    try:
        # Check if format is "Class:weight" or just "weight"
        if ':' in pairs[0]:
            # Named format: "Car:1.1,Human:0.5,..."
            for pair in pairs:
                cls, weight = pair.split(':')
                cls = cls.strip()
                class_weights[cls] = float(weight.strip())
        else:
            # Positional format: "1.1,0.5,1.8,1.8"
            weights = [float(w.strip()) for w in pairs]
            if len(weights) != len(class_names):
                print(f'Error: Expected {len(class_names)} weights, got {len(weights)}')
                return None
            class_weights = {cls: w for cls, w in zip(class_names, weights)}
    except (ValueError, IndexError) as e:
        print(f'Error parsing class weights: {e}')
        return None
    
    return class_weights if class_weights else None


def _get_pc_scope(class_list):
    # z_range covers both forward and backward to support 360-degree LiDAR.
    # With the dummy identity calibration, z_cam == x_lidar (forward axis), so
    # using [0, N] would silently discard all points behind the vehicle.
    if class_list == ['Car']:
        return np.array([[-40, 40], [-3, 3], [-70.4, 70.4]])
    return np.array([[-50, 50], [-3, 3], [-50, 50]])


def save_kitti_format(calib, bbox3d, obj_list, img_shape, save_fp):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)
    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    for k in range(bbox3d.shape[0]):
        if valid_mask[k] == 0:
            continue
        x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry
        print('%s %.2f %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
              (obj_list[k].cls_type, obj_list[k].trucation, int(obj_list[k].occlusion), alpha,
               img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
               bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2], bbox3d[k, 6]),
              file=save_fp)


class AugSceneGenerator(KittiDataset):
    def __init__(self, root_dir, gt_database, split, classes, use_weighted_sampling=True, manual_class_weights=None):
        super().__init__(root_dir, split=split)
        class_list = _parse_classes(classes)
        self.classes = tuple(['Background'] + class_list)
        self.pc_area_scope = _get_pc_scope(class_list)
        self.gt_database = gt_database
        self.use_weighted_sampling = use_weighted_sampling
        self.manual_class_weights = manual_class_weights  # Optional: manually specified weights
        self._setup_class_sampling()

    def _setup_class_sampling(self):
        """Set up weighted sampling based on class distribution in GT database or config"""
        if not self.use_weighted_sampling:
            self.class_weights = None
            self.class_indices = None
            return
        
        # Count objects per class in GT database
        class_counts = {}
        class_indices = {}
        for idx, gt_dict in enumerate(self.gt_database):
            cls_type = gt_dict['obj'].cls_type
            if cls_type not in class_counts:
                class_counts[cls_type] = 0
                class_indices[cls_type] = []
            class_counts[cls_type] += 1
            class_indices[cls_type].append(idx)
        
        # Determine which weights to use
        if self.manual_class_weights is not None:
            # Priority 1: Manual weights from command line
            class_weights = self.manual_class_weights
            print(f'Using manual class weights (from --class_weights):')
            for cls_type, weight in sorted(class_weights.items()):
                print(f'  {cls_type}: {weight:.4f}')
        else:
            # Priority 2: Config weights if available
            try:
                class_list = list(self.classes[1:])  # Skip 'Background'
                config_weights = cfg.RPN.CLS_WEIGHT[1:]  # Skip background weight at index 0
                
                if len(config_weights) == len(class_list):
                    class_weights = {cls: float(w) for cls, w in zip(class_list, config_weights)}
                    print(f'Using class weights from config (lib/config.py):')
                    for cls_type, weight in sorted(class_weights.items()):
                        print(f'  {cls_type}: {weight:.4f}')
                else:
                    raise ValueError(f'Config weights length mismatch: {len(config_weights)} vs {len(class_list)}')
            except (AttributeError, ValueError, TypeError):
                # Fallback: Auto-calculate from GT database
                total_count = sum(class_counts.values())
                class_weights = {}
                for cls_type, count in class_counts.items():
                    class_weights[cls_type] = total_count / (len(class_counts) * count)
                print(f'Using auto-calculated weights from GT database:')
                for cls_type, count in sorted(class_counts.items()):
                    print(f'  {cls_type}: {count} samples (weight: {class_weights[cls_type]:.4f})')
        
        self.class_counts = class_counts
        self.class_indices = class_indices
        self.class_weights = class_weights

    def _sample_gt_object(self):
        """Sample a GT object, with weighted sampling if enabled"""
        if not self.use_weighted_sampling or self.class_weights is None:
            # Uniform sampling (original behavior)
            return self.gt_database[np.random.randint(0, len(self.gt_database))]
        
        # Weighted sampling by class
        classes = list(self.class_weights.keys())
        weights = [self.class_weights[cls] for cls in classes]
        # Normalize weights to probabilities
        weights = np.array(weights) / sum(weights)
        
        # Sample a class based on weights
        selected_class = np.random.choice(classes, p=weights)
        # Sample a random object from the selected class
        class_idx_list = self.class_indices[selected_class]
        selected_idx = np.random.choice(class_idx_list)
        
        return self.gt_database[selected_idx]

    def filtrate_dc_objects(self, obj_list):
        return [o for o in obj_list if o.cls_type not in ['DontCare'] and o.cls_type.lower() not in ['box', 'boxes', 'bbox']]

    def filtrate_objects(self, obj_list):
        return [o for o in obj_list if o.cls_type in self.classes and o.cls_type.lower() not in ['box', 'boxes', 'bbox']]

    def get_valid_flag(self, pts_rect, pts_img, pts_rect_depth, img_shape):
        # Image-projection filtering is intentionally skipped: the custom dataset
        # uses a dummy identity calibration with no real camera image. Filtering on
        # pts_rect_depth >= 0 or image bounds would silently drop ~half the 360-deg scan.
        x_range, y_range, z_range = self.pc_area_scope
        pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
        return ((pts_x >= x_range[0]) & (pts_x <= x_range[1]) &
                (pts_y >= y_range[0]) & (pts_y <= y_range[1]) &
                (pts_z >= z_range[0]) & (pts_z <= z_range[1]))

    def check_pc_range(self, xyz):
        x_range, y_range, z_range = self.pc_area_scope
        return (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and (z_range[0] <= xyz[2] <= z_range[1])

    def aug_one_scene(self, sample_id, pts_rect, pts_intensity, all_gt_boxes3d):
        extra_gt_num = np.random.randint(10, 15)
        try_times = 50
        cnt = 0
        cur_gt_boxes3d = all_gt_boxes3d.copy()
        cur_gt_boxes3d[:, 4] += 0.5
        cur_gt_boxes3d[:, 5] += 0.5

        extra_gt_obj_list, extra_gt_boxes3d_list = [], []
        new_pts_list, new_pts_intensity_list = [], []
        src_pts_flag = np.ones(pts_rect.shape[0], dtype=np.int32)

        road_plane = self.get_road_plane(sample_id)
        a, b, c, d = road_plane

        while try_times > 0:
            try_times -= 1
            new_gt_dict = self._sample_gt_object()
            new_gt_box3d = new_gt_dict['gt_box3d'].copy()
            new_gt_points = new_gt_dict['points'].copy()
            new_gt_intensity = new_gt_dict['intensity'].copy()
            new_gt_obj = new_gt_dict['obj']
            center = new_gt_box3d[0:3]

            if not self.check_pc_range(center):
                continue
            if cnt > extra_gt_num:
                break

            cur_height = (-d - a * center[0] - c * center[2]) / b
            move_height = new_gt_box3d[1] - cur_height
            new_gt_box3d[1] -= move_height
            new_gt_points[:, 1] -= move_height
            cnt += 1

            iou3d = iou3d_utils.boxes_iou3d_gpu(torch.from_numpy(new_gt_box3d.reshape(1, 7)).cuda(),
                                                torch.from_numpy(cur_gt_boxes3d).cuda()).cpu().numpy()
            if iou3d.max() >= 1e-8:
                continue

            enlarged_box3d = new_gt_box3d.copy()
            enlarged_box3d[3] += 2
            pt_mask_flag = (roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(pts_rect), torch.from_numpy(enlarged_box3d.reshape(1, 7)))[0].numpy() == 1)
            src_pts_flag[pt_mask_flag] = 0

            new_pts_list.append(new_gt_points)
            new_pts_intensity_list.append(new_gt_intensity)
            enlarged_box3d = new_gt_box3d.copy()
            enlarged_box3d[4] += 0.5
            enlarged_box3d[5] += 0.5
            cur_gt_boxes3d = np.concatenate((cur_gt_boxes3d, enlarged_box3d.reshape(1, 7)), axis=0)
            extra_gt_boxes3d_list.append(new_gt_box3d.reshape(1, 7))
            extra_gt_obj_list.append(new_gt_obj)

        if len(new_pts_list) == 0:
            return False, pts_rect, pts_intensity, None, None

        extra_gt_boxes3d = np.concatenate(extra_gt_boxes3d_list, axis=0)
        pts_rect = pts_rect[src_pts_flag == 1]
        pts_intensity = pts_intensity[src_pts_flag == 1]
        pts_rect = np.concatenate((pts_rect, np.concatenate(new_pts_list, axis=0)), axis=0)
        pts_intensity = np.concatenate((pts_intensity, np.concatenate(new_pts_intensity_list, axis=0)), axis=0)
        return True, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list

    def generate_aug_scene(self, split, aug_times, save_dir, imagesets_dir):
        data_save_dir = os.path.join(save_dir, 'rectified_data')
        label_save_dir = os.path.join(save_dir, 'aug_label')
        os.makedirs(data_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        split_file = os.path.join(save_dir, f'{split}_aug.txt')
        split_list = self.image_idx_list.copy()

        for epoch in range(aug_times):
            base_id = (epoch + 1) * 10000
            for sample_id_str in self.image_idx_list:
                sample_id = int(sample_id_str)
                print(f'process gt sample ({split}, id={sample_id:06d})')

                pts_lidar = self.get_lidar(sample_id)
                calib = self.get_calib(sample_id)
                pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
                img_shape = self.get_image_shape(sample_id)

                valid = self.get_valid_flag(pts_rect, None, None, img_shape)
                pts_rect = pts_rect[valid][:, 0:3]
                pts_intensity = pts_lidar[valid][:, 3]

                all_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
                all_gt_boxes3d = np.zeros((len(all_obj_list), 7), dtype=np.float32)
                for k, obj in enumerate(all_obj_list):
                    all_gt_boxes3d[k, 0:3], all_gt_boxes3d[k, 3], all_gt_boxes3d[k, 4], all_gt_boxes3d[k, 5], all_gt_boxes3d[k, 6] = obj.pos, obj.h, obj.w, obj.l, obj.ry

                obj_list = self.filtrate_objects(self.get_label(sample_id))
                if len(obj_list) == 0:
                    continue

                aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = self.aug_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)

                out_id = base_id + sample_id
                np.concatenate((pts_rect, pts_intensity.reshape(-1, 1)), axis=1).astype(np.float32).tofile(
                    os.path.join(data_save_dir, f'{out_id:06d}.bin')
                )

                label_save_file = os.path.join(label_save_dir, f'{out_id:06d}.txt')
                with open(label_save_file, 'w') as f:
                    for obj in obj_list:
                        print(obj.to_kitti_format(), file=f)
                    if aug_flag:
                        save_kitti_format(calib, extra_gt_boxes3d, extra_gt_obj_list, img_shape=img_shape, save_fp=f)
                split_list.append(f'{out_id:06d}')

        with open(split_file, 'w') as f:
            f.write('\n'.join(split_list) + '\n')

        os.makedirs(imagesets_dir, exist_ok=True)
        target_file = os.path.join(imagesets_dir, os.path.basename(split_file))
        with open(split_file, 'r') as src, open(target_file, 'w') as dst:
            dst.write(src.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='generator')
    parser.add_argument('--class_name', type=str, default='Car,Human,ForkLift,CargoBike')
    parser.add_argument('--root_dir', type=str, default='../data/dataset')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--gt_database_dir', type=str, default='gt_database/train_gt_database_3level_multi.pkl')
    parser.add_argument('--aug_times', type=int, default=4)
    parser.add_argument('--skip_preprocess', action='store_true', default=False)
    parser.add_argument('--weighted_sampling', action='store_true', default=True, 
                        help='Use weighted sampling to balance class distribution during augmentation')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='Manual class weights. Format: "Car:1.1,Human:0.5,ForkLift:1.8,CargoBike:1.8" or "1.1,0.5,1.8,1.8"')
    args = parser.parse_args()

    root_dir_abs = _resolve_path(args.root_dir)
    class_list = _parse_classes(args.class_name)
    split_file = os.path.join(root_dir_abs, 'KITTI', 'ImageSets', f'{args.split}.txt')
    if not args.skip_preprocess and not os.path.exists(split_file):
        preprocess_dataset(root_dir_abs, classes=tuple(class_list))

    if args.save_dir is None:
        args.save_dir = os.path.join(root_dir_abs, 'KITTI', 'aug_scene', 'training')
    else:
        args.save_dir = _resolve_path(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    args.gt_database_dir = _resolve_path(args.gt_database_dir)
    if not os.path.exists(args.gt_database_dir):
        raise FileNotFoundError(f'gt_database file not found: {args.gt_database_dir}')

    # Parse manual class weights if provided
    manual_class_weights = None
    if args.class_weights:
        manual_class_weights = _parse_class_weights(args.class_weights, class_list)
        if manual_class_weights is None:
            print('Warning: Failed to parse class weights, using auto-calculated weights')

    gt_database = pickle.load(open(args.gt_database_dir, 'rb'))
    dataset = AugSceneGenerator(root_dir=root_dir_abs, gt_database=gt_database, split=args.split, 
                                classes=args.class_name, use_weighted_sampling=args.weighted_sampling,
                                manual_class_weights=manual_class_weights)
    imagesets_dir = os.path.join(root_dir_abs, 'KITTI', 'ImageSets')
    dataset.generate_aug_scene(args.split, args.aug_times, args.save_dir, imagesets_dir)


if __name__ == '__main__':
    main()
