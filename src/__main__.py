import fire
from tqdm import tqdm


class RAGPipeline:
    """
    RAG against the machine - CLI principal.
    """

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingere data/raw/ e constrói o índice sob data/processed/."""
        print(f"Iniciando indexação com max_chunk_size={max_chunk_size}...")
        # TODO: Implementar Fase 1 e 2

    def search(self, query: str, k: int) -> None:
        """Retorna as top-k fontes para uma única query."""
        print(f"Buscando '{query}' (Top {k})...")
        # TODO: Implementar Fase 2

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Roda a busca sobre um dataset completo e escreve um JSON."""
        print(f"Buscando dataset em {dataset_path}...")
        # TODO: Implementar Fase 2

    def answer(self, query: str, k: int) -> None:
        """Responde a uma única query usando o contexto recuperado."""
        print(f"Gerando resposta para '{query}'...")
        # TODO: Implementar Fase 3 e 4

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Gera respostas para um dataset baseado nos resultados de busca."""
        print(f"Gerando respostas baseadas em {student_search_results_path}...")
        # TODO: Implementar Fase 4

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """Avalia e reporta o recall@k (apenas para teste interno)."""
        print(f"Avaliando resultados de {student_search_results_path} contra {dataset_path}...")
        # TODO: Implementar avaliação básica


if __name__ == '__main__':
    fire.Fire(RAGPipeline)
