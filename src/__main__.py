import fire
from tqdm import tqdm

from src.indexer import ProjectIndexer

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
        print(f"Buscando '{query}' (Top {k})...")
        # TODO: Implementar Fase 2

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run the search over a complete dataset and write a JSON file."""
        print(f"Buscando dataset em {dataset_path}...")
        # TODO: Implementar Fase 2

    def answer(self, query: str, k: int) -> None:
        """Respond to a single query using the retrieved context."""
        print(f"Gerando resposta para '{query}'...")
        # TODO: Implementar Fase 3 e 4

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Generates responses for a dataset based on search results."""
        print(f"Gerando respostas baseadas em {student_search_results_path}...")
        # TODO: Implementar Fase 4

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """Evaluates and reports recall@k (for internal testing only)."""
        print(f"Avaliando resultados de {student_search_results_path} contra {dataset_path}...")
        # TODO: Implementar avaliação básica


if __name__ == '__main__':
    fire.Fire(RAGPipeline)
