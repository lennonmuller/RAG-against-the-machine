import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class AnswerGenerator:
    """
    Handles LLM inference to generate grounded answers using retrieved context.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        print(f"Loading LLM {model_name}... (This might take a moment to download weights)")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)  # type: ignore

        # Some Qwen models need pad_token defined
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _get_context_text(self, source: MinimalSource) -> str:
        """Reads the exact chunk from the raw file using character indices."""
        try:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content[source.first_character_index:source.last_character_index]
        except Exception as e:
            print(f"Warning: Failed to read context from {source.file_path}: {e}")
            return ""

    def _generate_single_answer(self, question: str, sources: List[MinimalSource]) -> str:
        """Constructs the prompt and calls the LLM for a single question."""

        # Build context from sources
        context_parts = []
        for src in sources:
            text = self._get_context_text(src)
            if text.strip():
                context_parts.append(f"Snippet from {src.file_path}:\n{text}")

        context_str = "\n\n".join(context_parts[:3])

        messages = [
            {"role": "system",
             "content": "You are a technical assistant. Answer strictly based on the context provided. Be brief."
             },
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
        ]

        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt", truncation=True, max_length=1500).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new generated tokens
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer_raw = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answer = str(answer_raw).strip()

        return answer

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Reads retrieved results, generates answers, and saves the final JSON."""
        in_path = Path(student_search_results_path)
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / in_path.name

        with open(in_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        search_results_data = StudentSearchResults.model_validate(raw_data)

        answered_results = []
        for result in tqdm(search_results_data.search_results, desc="Generating Answers"):
            answer_text = self._generate_single_answer(result.question, result.retrieved_sources)

            ans_obj = MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer_text
            )
            answered_results.append(ans_obj)

        final_output = StudentSearchResultsAndAnswer(
            search_results=answered_results,
            k=search_results_data.k
        )

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_output.model_dump_json(indent=2))

        print(f"Saved student_search_results_and_answer to {out_file}")
