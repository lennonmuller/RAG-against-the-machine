import sys

import fire

from src.evaluator import InternalEvaluator
from src.generator import AnswerGenerator
from src.indexer import ProjectIndexer
from src.retriever import BM25Retriever


class RAGPipeline:
    """
    RAG against the machine - CLI principal.
    """

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Ingests raw_dir and builds the index under processed_dir."""
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print("Error: max_chunk_size must be between 1 and 2000 characters.")
            return
        print(f"Starting indexing from '{raw_dir}' with max_chunk_size={max_chunk_size}...")
        indexer = ProjectIndexer(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_chunk_size=max_chunk_size,
        )
        indexer.run_ingestion()

    def search(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed/index.pkl",
    ) -> None:
        """Returns the top-k sources for a single query."""
        print(f"Searching '{query}' (Top {k})...")
        retriever = BM25Retriever(index_path=index_path)
        result = retriever.search_single(query_id="cli-manual", query=query, k=k)
        print(result.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        index_path: str = "data/processed/index.pkl",
    ) -> None:
        """Runs search over a whole dataset and writes a StudentSearchResults JSON file."""
        print(f"Searching dataset in {dataset_path}...")
        retriever = BM25Retriever(index_path=index_path)
        retriever.search_dataset(
            dataset_path=dataset_path, k=k, save_directory=save_directory
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed/index.pkl",
    ) -> None:
        """Answers a single query using the retrieved context."""
        print(f"Generating answer for '{query}'...")
        retriever = BM25Retriever(index_path=index_path)
        search_result = retriever.search_single(query_id="cli-manual", query=query, k=k)

        generator = AnswerGenerator()
        answer_text = generator._generate_single_answer(
            search_result.question, search_result.retrieved_sources
        )

        print("\n=== ANSWER ===")
        print(answer_text)
        print("==============\n")

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Generates responses for a dataset based on search results."""
        print(f"Generating answers based on {student_search_results_path}...")
        generator = AnswerGenerator()
        generator.answer_dataset(student_search_results_path, save_directory)

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """Evaluates and reports recall@k (for internal testing only)."""
        print(f"Evaluating results of {student_search_results_path} against {dataset_path}...")
        evaluator = InternalEvaluator()
        evaluator.evaluate(student_search_results_path, dataset_path)

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Starts the local HTTP API server using uvicorn."""
        import uvicorn

        print(f"Starting local HTTP API at http://{host}:{port}")
        uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == '__main__':
    try:
        fire.Fire(RAGPipeline)
    except ValueError as f:
        print(f"Error: {f}")
        sys.exit(1)
