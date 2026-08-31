import numpy as np
from scipy.optimize import linear_sum_assignment


class Stage2Evaluator:
    def evaluate_single(self, pred_instances, gt_instances):
        """Evaluate instance segmentation for one image.

        Args:
            pred_instances: int32 [H, W] predicted instance mask
            gt_instances: int32 [H, W] ground truth instance mask

        Returns:
            dict with count_error, precision, recall, f1, iou_per_instance
        """
        pred_ids = set(np.unique(pred_instances)) - {0}
        gt_ids = set(np.unique(gt_instances)) - {0}

        count_error = len(pred_ids) - len(gt_ids)

        if not gt_ids and not pred_ids:
            return {"count_error": 0, "precision": 1.0, "recall": 1.0,
                    "f1": 1.0, "iou_per_instance": {}}

        if not gt_ids or not pred_ids:
            return {"count_error": count_error, "precision": 0.0,
                    "recall": 0.0, "f1": 0.0, "iou_per_instance": {}}

        # Build IoU cost matrix
        pred_ids_list = sorted(pred_ids)
        gt_ids_list = sorted(gt_ids)
        cost_matrix = np.zeros((len(pred_ids_list), len(gt_ids_list)))

        for i, p in enumerate(pred_ids_list):
            pred_area = pred_instances == p
            for j, g in enumerate(gt_ids_list):
                gt_area = gt_instances == g
                intersection = (pred_area & gt_area).sum()
                union = (pred_area | gt_area).sum()
                cost_matrix[i, j] = intersection / (union + 1e-8)

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        tp = len(row_ind)
        matched_ious = cost_matrix[row_ind, col_ind]
        mean_iou = matched_ious.mean() if tp > 0 else 0.0

        precision = tp / len(pred_ids_list) if pred_ids_list else 0.0
        recall = tp / len(gt_ids_list) if gt_ids_list else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        iou_per_instance = {
            f"pred_{pred_ids_list[r]}_gt_{gt_ids_list[c]}": float(cost_matrix[r, c])
            for r, c in zip(row_ind, col_ind)
        }

        return {
            "count_error": count_error,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "iou_per_instance": iou_per_instance,
        }

    def evaluate_batch(self, results, gt_masks):
        """Evaluate a batch of predictions against ground truth masks.

        Args:
            results: list of dicts from Stage2Pipeline.process()
            gt_masks: list of int32 [H, W] ground truth instance masks

        Returns:
            dict with aggregate metrics
        """
        all_metrics = []
        for result, gt in zip(results, gt_masks):
            metrics = self.evaluate_single(result["instance_mask"], gt)
            all_metrics.append(metrics)

        count_errors = [m["count_error"] for m in all_metrics]
        precisions = [m["precision"] for m in all_metrics]
        recalls = [m["recall"] for m in all_metrics]
        f1s = [m["f1"] for m in all_metrics]
        mean_ious = [m.get("mean_iou", 0.0) for m in all_metrics]

        return {
            "mean_count_error": float(np.mean(count_errors)),
            "mean_precision": float(np.mean(precisions)),
            "mean_recall": float(np.mean(recalls)),
            "mean_f1": float(np.mean(f1s)),
            "mean_iou": float(np.mean(mean_ious)),
            "per_image": all_metrics,
        }
