from typing import List, Tuple
from llmlib.vectorestore_components import CustomChromaDB


class BaseRetriever(object):
    """
    Base retriever class to be inherited by other retriever classes to perform context retrieval process.
    
    The Retriever should use the same embedding model and its tokenizer which is used during the ingestion process.
    """

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 db_path: str,
                 collection_name: str) -> None:
        """
        Method to instantiate object of :class: Retriever.

        :Parameters:
        model_path: Absolute path to the embedding model to be used for retrieval
        tokenizer_path: Absolute path to the tokenizer model corresponding to the embedding model to be used for
                        retrieval
        db_path: Absolute path to the vector store
        collection_name: Name of the collection to query from

        :Returns:
        None
        """

        super().__init__()

        self._chroma_collection = CustomChromaDB(model_path=model_path,
                                                 tokenizer_path=tokenizer_path,
                                                 db_path=db_path,
                                                 collection_name=collection_name)

    def query(self) -> None:
        """
        Method to implement retrieval of documents from the collection which each subclass of
        :class: Retrieval implements on its own.
        """

        pass

    def query_with_distance(self) -> None:
        """
        Method to implement retrieval of documents from the collection with their respective distance from
        the query provided by the user which each subclass of :class: Retriever needs to implement on its own.
        """

        pass


class Retriever(BaseRetriever):
    """
    Class to implement a simple retriever process
    
    """

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 db_path: str,
                 collection_name: str,
                 top_k: int) -> None:

        """
        Method to instantiate object of :class: PDFRetriever

        :Parameters:
        model_path: Absolute path to the embedding model to be used for retrieval
        tokenizer_path: Absolute path to the tokenizer model corresponding to the embedding model 
                        to be used for retrieval
        db_path: Path to the database that contains all vector databases
        collection_name: Name of the collection to query from
        top_k: Top K documents to retrieve from the collection

        :Returns:
        None
        """

        super().__init__(model_path=model_path,
                         tokenizer_path=tokenizer_path,
                         db_path=db_path,
                         collection_name=collection_name)
        
        self._top_k = top_k

    def query(self, query: str) -> List[str]:
        """
        Method to implement querying of top k documents with respect to the query asked by the user.

        The query will be a single string for which 'top k' documents will be retrieved based on their
        similarity match.

        :Parameters:
        query: Query string for which documents are to be retrieved.

        :Returns:
        List of documents retrieved from the collection based on similarity match with the user query.
        """

        # put the query into a list to be passed into the chroma db collection
        retrieved_documents = self._chroma_collection.get_documents(query=[query], top_k=self._top_k)

        return retrieved_documents
    
    def query_with_distance(self, query: str) -> Tuple[List[str], List[float]]:
        """
        Method to implement querying of top k documents with respect to the query asked by the user.
        This method will also return the distance of the retrieved documents with respect to the query
        
        The query will be a single string for which 'top k' documents will be retrieved based on their similarity match.

        :Parameters:
        query: Query string for which documents are to be retrieved.

        :Returns:
        Tuple containing lists of documents and their distance retrieved from the collection based on the
        similarity match with the user query
        """
        
        retrieved_documents, retrieved_distances = self._chroma_collection.get_documents_with_distance(query=[query],
                                                                                                       top_k=self._top_k)
        
        return retrieved_documents, retrieved_distances