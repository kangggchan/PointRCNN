import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.bbox_transform import decode_bbox_target
from lib.config import cfg
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils


class ProposalLayer(nn.Module):
    def __init__(self, mode='TRAIN'):
        super().__init__()
        self.mode = mode
        self.mean_sizes = torch.from_numpy(kitti_utils.get_mean_size_array()).float()
        self.num_size_templates = self.mean_sizes.shape[0]

    def get_point_cls_info(self, rpn_cls):
        if rpn_cls.dim() == 3 and rpn_cls.shape[2] > 1:
            cls_probs = F.softmax(rpn_cls, dim=2)
            pred_classes = torch.argmax(cls_probs, dim=2)
            point_probs, template_ids = torch.max(cls_probs[:, :, 1:], dim=2)
            point_scores = torch.log(point_probs.clamp(min=1e-6)) - torch.log((1.0 - point_probs).clamp(min=1e-6))
            fg_mask = pred_classes > 0
            return point_scores, point_probs, template_ids, pred_classes, fg_mask

        point_scores = rpn_cls[:, :, 0] if rpn_cls.dim() == 3 else rpn_cls
        point_probs = torch.sigmoid(point_scores)
        fg_mask = torch.ones_like(point_probs, dtype=torch.bool)
        return point_scores, point_probs, None, None, fg_mask

    def forward(self, rpn_cls, rpn_reg, xyz, return_class_ids=False):
        """
        :param rpn_cls: (B, N) or (B, N, C)
        :param rpn_reg: (B, N, C)
        :param xyz: (B, N, 3)
        :return bbox3d: (B, M, 7)
        """
        batch_size = xyz.shape[0]
        flat_xyz = xyz.view(-1, 3)
        flat_reg = rpn_reg.view(-1, rpn_reg.shape[-1])
        point_scores, _, template_ids, pred_classes, fg_mask = self.get_point_cls_info(rpn_cls)
        proposal_class_ids = None

        if template_ids is None:
            proposal_list = []
            class_id_list = [] if return_class_ids else None
            for template_idx in range(self.num_size_templates):
                cur_template_ids = torch.full((flat_xyz.shape[0],), template_idx, dtype=torch.long, device=xyz.device)
                template_proposals = decode_bbox_target(
                    flat_xyz,
                    flat_reg,
                    anchor_size=self.mean_sizes[template_idx].to(xyz.device),
                    loc_scope=cfg.RPN.LOC_SCOPE,
                    loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                    num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                    get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                    get_y_by_bin=False,
                    get_ry_fine=False,
                    size_template_ids=cur_template_ids,
                )
                template_proposals[:, 1] += template_proposals[:, 3] / 2  # set y as the center of bottom
                proposal_list.append(template_proposals.view(batch_size, -1, 7))
                if return_class_ids:
                    class_id_list.append(torch.full((batch_size, xyz.shape[1]), template_idx + 1,
                                                    dtype=torch.long, device=xyz.device))

            proposals = torch.cat(proposal_list, dim=1)
            scores = point_scores.repeat(1, self.num_size_templates)
            if return_class_ids:
                proposal_class_ids = torch.cat(class_id_list, dim=1)
        else:
            flat_template_ids = template_ids.view(-1)
            anchor_sizes = kitti_utils.get_anchor_sizes_by_template_ids_torch(flat_template_ids, device=xyz.device)
            proposals = decode_bbox_target(
                flat_xyz,
                flat_reg,
                anchor_size=anchor_sizes,
                loc_scope=cfg.RPN.LOC_SCOPE,
                loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                get_y_by_bin=False,
                get_ry_fine=False,
                size_template_ids=flat_template_ids,
            )
            proposals[:, 1] += proposals[:, 3] / 2  # set y as the center of bottom
            proposals = proposals.view(batch_size, -1, 7)
            scores = point_scores
            if return_class_ids:
                proposal_class_ids = pred_classes

        batch_size = scores.size(0)
        ret_bbox3d = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N, 7).zero_()
        ret_scores = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N).zero_()
        ret_class_ids = torch.zeros((batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N), dtype=torch.long,
                                    device=xyz.device) if return_class_ids else None
        for k in range(batch_size):
            if template_ids is None:
                scores_single = scores[k]
                proposals_single = proposals[k]
                class_ids_single = proposal_class_ids[k] if return_class_ids else None
            else:
                candidate_mask = fg_mask[k]
                scores_single = scores[k][candidate_mask]
                proposals_single = proposals[k][candidate_mask]
                class_ids_single = proposal_class_ids[k][candidate_mask] if return_class_ids else None

            if scores_single.shape[0] == 0:
                continue

            _, order_single = torch.sort(scores_single, dim=0, descending=True)

            if cfg[self.mode].RPN_DISTANCE_BASED_PROPOSE:
                scores_single, proposals_single, class_ids_single = self.distance_based_proposal(
                    scores_single,
                    proposals_single,
                    order_single,
                    class_ids_single,
                )
            else:
                scores_single, proposals_single, class_ids_single = self.score_based_proposal(
                    scores_single,
                    proposals_single,
                    order_single,
                    class_ids_single,
                )

            scores_single, proposals_single, class_ids_single = self.filter_empty_proposals(
                xyz[k:k + 1],
                scores_single,
                proposals_single,
                class_ids_single,
            )

            proposals_tot = proposals_single.size(0)
            ret_bbox3d[k, :proposals_tot] = proposals_single
            ret_scores[k, :proposals_tot] = scores_single
            if return_class_ids and proposals_tot > 0:
                ret_class_ids[k, :proposals_tot] = class_ids_single

        if return_class_ids:
            return ret_bbox3d, ret_scores, ret_class_ids

        return ret_bbox3d, ret_scores

    def filter_empty_proposals(self, xyz, scores, proposals, class_ids=None):
        if proposals.shape[0] == 0 or not cfg.RPN.FILTER_EMPTY_PROPOSALS:
            return scores, proposals, class_ids

        pooled_features, pooled_empty_flag = roipool3d_utils.roipool3d_gpu(
            xyz,
            xyz.new_zeros((xyz.shape[0], xyz.shape[1], 1)),
            proposals.unsqueeze(0),
            cfg.RCNN.POOL_EXTRA_WIDTH,
            sampled_pt_num=1,
        )
        del pooled_features

        valid_mask = pooled_empty_flag[0] == 0
        if valid_mask.any():
            if class_ids is None:
                return scores[valid_mask], proposals[valid_mask], None
            return scores[valid_mask], proposals[valid_mask], class_ids[valid_mask]

        empty_class_ids = class_ids[:0] if class_ids is not None else None
        return scores[:0], proposals[:0], empty_class_ids

    def distance_based_proposal(self, scores, proposals, order, class_ids=None):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        nms_range_list = [0, 40.0, 80.0]
        pre_tot_top_n = cfg[self.mode].RPN_PRE_NMS_TOP_N
        pre_top_n_list = [0, int(pre_tot_top_n * 0.7), pre_tot_top_n - int(pre_tot_top_n * 0.7)]
        post_tot_top_n = cfg[self.mode].RPN_POST_NMS_TOP_N
        post_top_n_list = [0, int(post_tot_top_n * 0.7), post_tot_top_n - int(post_tot_top_n * 0.7)]

        scores_single_list, proposals_single_list = [], []
        class_ids_single_list = [] if class_ids is not None else None

        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]
        class_ids_ordered = class_ids[order] if class_ids is not None else None

        dist = proposals_ordered[:, 2]
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])
        second_mask = (dist > nms_range_list[1]) & (dist <= nms_range_list[2])

        if first_mask.sum() == 0 or second_mask.sum() == 0:
            return self.score_based_proposal(scores, proposals, order, class_ids)

        for i in range(1, len(nms_range_list)):
            # get proposal distance mask
            dist_mask = ((dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i]))

            if dist_mask.sum() != 0:
                # this area has points
                # reduce by mask
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]
                cur_class_ids = class_ids_ordered[dist_mask] if class_ids_ordered is not None else None

                # fetch pre nms top K
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
                if cur_class_ids is not None:
                    cur_class_ids = cur_class_ids[:pre_top_n_list[i]]
            else:
                # this area doesn't have any points, so use rois of first area
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]
                cur_class_ids = class_ids_ordered[first_mask] if class_ids_ordered is not None else None

                # fetch top K of first area
                start_idx = min(pre_top_n_list[i - 1], cur_scores.shape[0])
                cur_scores = cur_scores[start_idx:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[start_idx:][:pre_top_n_list[i]]
                if cur_class_ids is not None:
                    cur_class_ids = cur_class_ids[start_idx:][:pre_top_n_list[i]]

            if cur_proposals.shape[0] == 0:
                continue

            # oriented nms
            boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
            if cfg.RPN.NMS_TYPE == 'rotate':
                keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            elif cfg.RPN.NMS_TYPE == 'normal':
                keep_idx = iou3d_utils.nms_normal_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
            else:
                raise NotImplementedError

            # Fetch post nms top k
            keep_idx = keep_idx[:post_top_n_list[i]]

            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])
            if class_ids_single_list is not None:
                class_ids_single_list.append(cur_class_ids[keep_idx])

        if len(scores_single_list) == 0:
            empty_scores = scores.new_zeros((0,))
            empty_proposals = proposals.new_zeros((0, proposals.shape[1]))
            empty_class_ids = class_ids.new_zeros((0,), dtype=torch.long) if class_ids is not None else None
            return empty_scores, empty_proposals, empty_class_ids

        scores_single = torch.cat(scores_single_list, dim=0)
        proposals_single = torch.cat(proposals_single_list, dim=0)
        class_ids_single = torch.cat(class_ids_single_list, dim=0) if class_ids_single_list is not None else None
        return scores_single, proposals_single, class_ids_single

    def score_based_proposal(self, scores, proposals, order, class_ids=None):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]
        class_ids_ordered = class_ids[order] if class_ids is not None else None

        # pre nms top K
        cur_scores = scores_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_proposals = proposals_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N]
        cur_class_ids = class_ids_ordered[:cfg[self.mode].RPN_PRE_NMS_TOP_N] if class_ids_ordered is not None else None

        if cur_proposals.shape[0] == 0:
            empty_class_ids = class_ids.new_zeros((0,), dtype=torch.long) if class_ids is not None else None
            return cur_scores, cur_proposals, empty_class_ids

        boxes_bev = kitti_utils.boxes3d_to_bev_torch(cur_proposals)
        if cfg.RPN.NMS_TYPE == 'rotate':
            keep_idx = iou3d_utils.nms_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
        elif cfg.RPN.NMS_TYPE == 'normal':
            keep_idx = iou3d_utils.nms_normal_gpu(boxes_bev, cur_scores, cfg[self.mode].RPN_NMS_THRESH)
        else:
            raise NotImplementedError

        # Fetch post nms top k
        keep_idx = keep_idx[:cfg[self.mode].RPN_POST_NMS_TOP_N]

        ret_class_ids = cur_class_ids[keep_idx] if cur_class_ids is not None else None
        return cur_scores[keep_idx], cur_proposals[keep_idx], ret_class_ids



