import json
from pathlib import Path
from src.models import MinimalSource, RagDataset, StudentSearchResults


class InternalEvaluator:
    """
    Calculates internal recall@k to satisfy the CLI evaluate command requirement.
    Does not rely on the external moulinette.
    """

    def _check_overlap(self, src_true: MinimalSource, src_pred: MinimalSource) -> bool:
        """
        Checks if two sources are from the same file and have an IoU > 0.05.
        """
        if src_true.file_path != src_pred.file_path:
            return False

        start_max = max(src_true.first_character_index, src_pred.first_character_index)
        end_min = min(src_true.last_character_index, src_pred.last_character_index)
        intersection = max(0, end_min - start_max)

        len_true = src_true.last_character_index - src_true.first_character_index
        len_pred = src_pred.last_character_index - src_true.first_character_index
        union = len_true + len_pred - intersection

        if union == 0:
            return False

        iou = intersection / union

        return iou >= 0.05

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """
        Loads the student results and ground truth, calculates and prints the recall.
        """
        ...
