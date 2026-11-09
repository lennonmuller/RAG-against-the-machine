import json

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
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            student_data = StudentSearchResults.model_validate(json.load(f))

        with open(dataset_path, "r", encoding="utf-8") as f:
            truth_data = RagDataset.model_validate(json.load(f))

        total_questions = len(truth_data.rag_questions)
        if total_questions == 0:
            print("Ground truth dataset is empty.")
            return

        student_results_dict = {res.question_id: res for res in student_data.search_results}

        hits = 0

        for truth_q in truth_data.rag_questions:
            if not hasattr(truth_q, 'sources') or not truth_q.sources:
                continue

            student_q = student_results_dict.get(truth_q.question_id)
            if not student_q:
                continue

            question_hit = False
            for true_src in truth_q.sources:
                for pred_src in student_q.retrieved_sources:
                    if self._check_overlap(true_src, pred_src):
                        question_hit = True
                        break
                if question_hit:
                    break

            if question_hit:
                hits += 1

        recall = hits / total_questions
        print(40*"=")
        print("Internal Evaluation Results:")
        print(f"Questions evaluated: {total_questions}")
        print(f"Recall@{student_data.k}: {recall:.3f} ({recall * 100:.1f}%)")
        print(40*"=")
