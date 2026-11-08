import fire

from src.evaluator import InternalEvaluator
from src.generator import AnswerGenerator
from src.indexer import ProjectIndexer
from src.retriever import BM25Retriever


class RAGPipeline:
    """
    RAG against the machine - CLI principal.
    """

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingests data/raw/ and builds the index under data/processed/."""
        print(f"Starting indexing with max_chunk_size={max_chunk_size}...")
        indexer = ProjectIndexer(max_chunk_size=max_chunk_size)
        indexer.run_ingestion()

    def search(self, query: str, k: int) -> None:
        """Returns the top-k sources for a single query."""
        print(f"Searching '{query}' (Top {k})...")
        retriever = BM25Retriever()
        result = retriever.search_single(query_id="cli-manual", query=query, k=k)
        print(result.model_dump_json(indent=2))

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run the search over a complete dataset and write a JSON file."""
        print(f"Searching dataset in {dataset_path}...")
        retriever = BM25Retriever()
        retriever.search_dataset(dataset_path=dataset_path, k=k, save_directory=save_directory)

    def answer(self, query: str, k: int) -> None:
        """Respond to a single query using the retrieved context."""
        print(f"Generating answer for '{query}'...")
        # 1. Retrieve
        from src.retriever import BM25Retriever
        retriever = BM25Retriever()
        search_result = retriever.search_single(query_id="cli-manual", query=query, k=k)

        generator = AnswerGenerator()
        answer_text = generator._generate_single_answer(search_result.question, search_result.retrieved_sources)

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


if __name__ == '__main__':
    fire.Fire(RAGPipeline)
