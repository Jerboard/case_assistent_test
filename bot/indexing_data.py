from objects import EoraCaseIndexer
from config import conf


if __name__ == "__main__":
    # если папка пустая, то проводим индексацию
    if not any(conf.persist_dir.iterdir()):
        indexer = EoraCaseIndexer(source_urls_path=conf.source_urls_path, vector_data_dir=conf.persist_dir)
        indexer.fetch_pages()