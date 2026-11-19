import json

from src.models import AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults


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
        len_pred = src_pred.last_character_index - src_pred.first_character_index

        if len_true <= 0 or len_pred <= 0:
            return False

        union = len_true + len_pred - intersection
        if union <= 0:
            return False

        iou = intersection / union
        return iou >= 0.05

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """
        Loads the student results and ground truth, calculates and prints the recall.
        """
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_data = StudentSearchResults.model_validate(json.load(f))

            with open(dataset_path, "r", encoding="utf-8") as f:
                truth_data = RagDataset.model_validate(json.load(f))
        except FileNotFoundError as e:
            print(f"Error: File not found - {e.filename}")
            return
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error: Failed to load or validate JSON - {e}")
            return

        total_questions = len(truth_data.rag_questions)
        if total_questions == 0:
            print("Ground truth dataset is empty.")
            return

        student_results_dict = {res.question_id: res for res in student_data.search_results}

        evaluated_questions = 0

        k_values = [val for val in [1, 3, 5, 10] if val <= student_data.k]
        if student_data.k > 0 and student_data.k not in k_values:
            k_values.append(student_data.k)
            k_values.sort()

        if not k_values:
            print("k must be greater than 0 to compute recall.")
            return

        recall_sums = {k_val: 0.0 for k_val in k_values}
        evaluated_questions = 0

        for truth_q in truth_data.rag_questions:
            if not isinstance(truth_q, AnsweredQuestion) or not truth_q.sources:
                continue

            evaluated_questions += 1
            student_q = student_results_dict.get(truth_q.question_id)
            if not student_q:
                continue

            for k_val in k_values:
                top_k_preds = student_q.retrieved_sources[:k_val]
                found_sources = sum(
                    1
                    for true_src in truth_q.sources
                    if any(
                        self._check_overlap(true_src, pred_src)
                        for pred_src in top_k_preds
                    )
                )
                recall_sums[k_val] += found_sources / len(truth_q.sources)

        if evaluated_questions == 0:
            print("No answered questions with sources found in ground truth.")
            return

        print(40*"=")
        print("Internal Evaluation Results")
        print(f"Questions evaluated: {evaluated_questions}")
        for k_val in k_values:
            avg_recall = recall_sums[k_val] / evaluated_questions
            print(f"Recall@{k_val}: {avg_recall:.3f} ({avg_recall * 100:.1f}%)")
        print(40*"=")
