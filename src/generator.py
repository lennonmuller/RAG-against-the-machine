import json
from pathlib import Path
from typing import Any, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cache import QueryCache
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

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        print(f"Loading LLM {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model: Any = loaded_model.to(self.device)

        # Clear sampling params to avoid warnings when using greedy decoding
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

        self.cache = QueryCache()

    def _get_context_text(self, source: MinimalSource) -> str:
        """Reads the exact chunk from the raw file using character indices."""
        try:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            start = max(0, source.first_character_index)
            end = max(start, source.last_character_index)
            return content[start:end]
        except Exception as e:
            print(f"Warning: Failed to read {source.file_path}: {e}")
            return ""

    def _build_prompt(
        self, question: str, sources: List[MinimalSource]
    ) -> str:
        """Constructs a compact, grounded chat prompt for the LLM."""
        context_parts: List[str] = []
        for src in sources[:2]:
            text = self._get_context_text(src).strip()
            if text:
                context_parts.append(
                    f"Snippet from {src.file_path}:\n{text[:1200]}"
                )

        context_str = "\n\n".join(context_parts)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise technical assistant. "
                    "Answer strictly based on the provided context."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {question}",
            },
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_str = str(prompt)
        if "</think>" not in prompt_str:
            prompt_str += "<think>\n\n</think>\n\n"
        return prompt_str

    def _clean_output(self, text: str) -> str:
        """Strips any residual thinking tags from generated output."""
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        elif "<think>" in text:
            text = text.replace("<think>", "")
        return text.strip()

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Runs batched LLM generation on a list of formatted prompts."""
        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(self.device)

        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        answers: List[str] = []
        for out in outputs:
            generated_tokens = out[input_length:]
            raw_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            answers.append(self._clean_output(raw_text))
        return answers

    def _generate_single_answer(
        self, question: str, sources: List[MinimalSource]
    ) -> str:
        """Generates an answer for a single question."""
        if not sources:
            return "No relevant context was retrieved to answer this question."
        prompt = self._build_prompt(question, sources)
        return self._generate_batch([prompt])[0]

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        batch_size: int = 4,
    ) -> None:
        """Reads search results, generates answers in batches, saves JSON."""
        in_path = Path(student_search_results_path)
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / in_path.name

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            search_results_data = StudentSearchResults.model_validate(raw_data)
        except FileNotFoundError:
            print(f"Error: File not found - {in_path}")
            return
        except Exception as e:
            print(f"Error: Failed to load search results JSON - {e}")
            return

        results = search_results_data.search_results
        k = search_results_data.k
        answered_results: List[MinimalAnswer] = []

        for i in tqdm(
            range(0, len(results), batch_size), desc="Generating Answers"
        ):
            batch = results[i:i + batch_size]
            uncached_indices: List[int] = []
            uncached_prompts: List[str] = []
            batch_answers: List[str] = [""] * len(batch)

            for idx, item in enumerate(batch):
                if not item.retrieved_sources:
                    batch_answers[idx] = (
                        "No relevant context was retrieved to answer this."
                    )
                    continue

                cached = self.cache.get("answer", item.question, k)
                if isinstance(cached, str):
                    batch_answers[idx] = cached
                else:
                    uncached_indices.append(idx)
                    uncached_prompts.append(
                        self._build_prompt(
                            item.question, item.retrieved_sources
                        )
                    )

            if uncached_prompts:
                generated = self._generate_batch(uncached_prompts)
                for pos, ans_text in zip(uncached_indices, generated):
                    batch_answers[pos] = ans_text
                    self.cache.set("answer", batch[pos].question, k, ans_text)

            for item, ans_text in zip(batch, batch_answers):
                answered_results.append(
                    MinimalAnswer(
                        question_id=item.question_id,
                        question=item.question,
                        retrieved_sources=item.retrieved_sources,
                        answer=ans_text,
                    )
                )

        final_output = StudentSearchResultsAndAnswer(
            search_results=answered_results,
            k=k,
        )

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_output.model_dump_json(indent=2))

        print(f"Saved student_search_results_and_answer to {out_file}")
