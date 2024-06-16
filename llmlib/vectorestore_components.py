import torch
import chromadb
from string import Template
from transformers import AutoTokenizer
from typing import Tuple, Sequence, List
from llmlib.embedding_configs import get_embedding_config
from chromadb import Documents, EmbeddingFunction, Embeddings, ClientAPI, Collection


class CustomChromaEmbeddings(EmbeddingFunction):
    """
    Class to implement custom embedding function to be used with chroma vector database.

    This class works with pre loaded embedding model and tokenizers. This class uses mean pooling on the output of the
    embedding model based on the paper `Sentence Embeddings using Siamese BERT-Networks`
    """

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        """
        Method to instantiate object of :class: CustomChromaEmbeddings

        :Parameters:
        model_path: Path to the locally saved embedding model to be used in chromadb
        tokenizer_path: path to the locally saved tokenizer model corresponding to the embedding model

        :Returns:
        None
        """

        super().__init__()

        self._model_path = model_path
        self._tokenizer_path = tokenizer_path

        # Load locally saved embedding model and its tokenizer
        self.model, self.tokenizer = get_embedding_config(model_path=self._model_path,
                                                          tokenizer_path=self._tokenizer_path)
        # Switch the model to evaluation mode
        self.model.eval()

    def __call__(self, documents: Documents) -> Embeddings:
        """
        Method to use the object of :class: CustomChromaEmbeddings as a callable function to convert documents into
        usable embeddings. The class currently works with text modality only.

        :Parameters:
        documents: List of text ot be converted into embeddings.

        :Returns:
        List of embeddings correpsonding to the documents.
        """

        # Instantiate an empty list to store document's embeddings
        document_embeddings = list()

        # Loop through each document and convert them into their embeddings
        for document in documents:
            tokenizer_output = self.tokenizer(document, padding=True, return_tensors='pt')

            token_ids, attention_mask = tokenizer_output['input_ids'], tokenizer_output['attention_mask']

            # Get all token embeddings from the model
            token_embeddings = self.model.forward(token_ids, attention_mask)[0]
            # Using mean pooling convert token embeddings into normalized sentence embeddings
            document_embedding = self.mean_pooling(embeddings=token_embeddings, attention_mask=attention_mask)
            # Store newly created document embedding into the list of document embeddings
            document_embeddings.append(document_embedding)

        return document_embeddings
    
    @staticmethod
    def mean_pooling(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Sequence[float]:
        """
        Function to convert output embeddings form the embedding model into sentence embeddings.

        The process of mean pooling as follows:
        output embeddings -> output embeddings (attention aware) -> mean pooling of embeddings -> sentence embeddings
        -> normalized sentence embeddings

        :Parameters:
        embeddings: Embeddings obtained from the underlying embedding model
        attention_mask: Attention mask corresponding to the embeddings

        :Returns:
        List of normalized sentence embeddings corresponding to the documents provided
        """

        # Convert attention mask from  the shape of (N, S) to (N, S, 1) shape
        new_attention_mask = attention_mask.unsqueeze(dim=-1).expand(size=embeddings.size()).float()
        # Convert embeddings into attention aware embeddings
        embeddings = embeddings * new_attention_mask
        # Convert embeddings itno sentence embeddings
        embeddings = torch.sum(embeddings, dim=1) / new_attention_mask.sum(dim=1)
        # Making sure there is no underflow
        embeddings = torch.clamp(embeddings, min=1e-9)
        # Convert embeddings from a batch tensor to a single embedding tensor
        embeddings = embeddings.squeeze(dim=0)
        # Normalize the sentence embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().tolist()


class CustomChromaDB(object):
    """
    Class that handles interaction with chromadb and its collections which are persisted on the local device. This
    class uses cosine similarity for similarity match. Embeddings of each chunk is marked with an id and each chunk
    is stored to its source name provided in the meta data. 
    
    Functionality to dynamically store documents is not currently available.
    Functionality to know which all documents exist within a collection is not currently available.
    Functionality to asynchronously add multiple documents is not currently available.
    """

    def __init__(self, model_path: str, tokenizer_path: str, db_path: str, collection_name: str) -> None:
        """
        Method to instantiate an object of :class: CustomChromaDB

        :Parameters:
        model_path: Path to the embedding model to be used while retrieval and ingestion of documents
        tokenizer_path: Path to the embedding tokenizer to be used while retrieval and ingestion of documents
        db_path: Path to the database where all of the collections are stored
        collection_name: Name of the collection from which retrieval or ingestion of vector needs to be done

        :Returns:
        None
        """

        super().__init__()

        self._collection_name = collection_name

        self.embedding_function = CustomChromaEmbeddings(model_path=model_path, tokenizer_path=tokenizer_path)
        self.persistent_client = self._create_client(db_path=db_path)
        self.collection = self._get_or_create_collection(collection_name=collection_name)
        self.id_template = Template('id: ${id}')

    def add_documents(self, documents: List[str], source_file_path: str) -> bool:
        """
        Method to add documents to the current chromadb collection. Documents refer to the list of chunks of text
        which are obtained from chunking source file.  

        :Parameters:
        documents: documents to be added to the chroma db collection
        source_file_path: Path to the main file from which the document has been selected

        :Returns:
        Boolean representing the successful completion of addition of the documents
        """

        # To add more documents to a collection we need to assign it a unique id. In order to get the unqiue id we
        # to count how many documents already exist within current collection.
        existing_length = self.collection.count()
        # Add the new documents into the collection
        self.collection.add(ids=[self.id_template.safe_substitute({'id': i}) for i in range(existing_length + 1, existing_length + len(documents) + 1)],
                            documents=documents,
                            metadatas=[{'source': source_file_path} for _ in range(len(documents))])
        
        return True
    
    def get_documents(self, query: List[str], top_k: int) -> List[str]:
        """
        Method to retrieve top k documents related ot the query.

        :Parameters:
        query: Query for which documents are to be retrieved
        top_k: value to specify how many documents are to be retrieved

        :Returns:
        List of top k match documents
        """

        # Get the query results from the chroma database
        query_result = self.collection.query(query_texts=query, n_results=top_k)

        documents = query_result['documents'][0]

        return documents
    
    def get_documents_with_distance(self, query: List[str], top_k: int) -> Tuple[List[str], List[float]]:
        """
        Method to retrieve top k documents related ot the query.

        :Parameters:
        query: Query for which documents are to be retrieved
        top_k: value to specify how many documents are to be retrieved

        :Returns:
        List of top k match documents
        """
        # Get the query results from the chroma database
        query_result = self.collection.query(query_texts=query, n_results=top_k)

        documents, distances = query_result['documents'][0], query_result['distances'][0]

        return documents, distances

    def delete_collection(self, collection_name: str) -> bool:
        """
        method to delete any existing collection given its name

        :Parameters:
        collection_name: Name of the collection that needs to be deleted

        :Returns:
        A boolean representing the completion of the process
        """

        self.persistent_client.delete_collection(name=collection_name)

        return True

    def delete_current_collection(self) -> bool:
        """
        Method to delete current collection

        :Parameters:
        none

        :Returns:
        A boolean representing the completion of the process
        """

        return self.delete_collection(collection_name=self._collection_name)
    
    def list_existing_collections(self) -> List[str]:
        """
        Method to return the list of all existing collections present in the current vector store.

        :Parameters:
        None:

        Returns:
        A list of names of all collections current present in the current vector store.
        """

        return [collection.name for collection in self.persistent_client.list_collections()]

    def source_documents_exists(self, source_file_path: str) -> bool:
        """
        Public method to use and check if the source document exists withing the loaded collection.

        :Parameters:
        source_file_path: Path to the document which is being ingested.

        :Returns:
        A boolean value indicating the existence of the document within the collection
        """

        return self._source_document_exists(source_file_path=source_file_path)

    def _create_client(self, db_path: str) -> ClientAPI:
        """
        Method to create a chroma db persistent client

        :Parameters:
        db_path: Path to the database where all of the collections are stored

        :Returns:
        Persistent chroma db client
        """

        return chromadb.PersistentClient(path=db_path)
    
    def _get_or_create_collection(self, collection_name: str) -> Collection:
        """
        Method to create a collection using a persistent chromadb client

        :Parameters:
        collection_name: Name of the collection to be loaded

        :Returns:
        Chromadb collection in which vector embeddings are stored
        """

        return self.persistent_client.get_or_create_collection(name=collection_name,
                                                               embedding_function=self.embedding_function)
    
    def _source_document_exists(self, source_file_path: str) -> bool:
        """
        Method to check if the source document already exists withing the loaded collection

        :Parameters:
        source_file_path: Path to the document which is being ingested

        :Returns:
        A boolean value indicating the existence of the document within the collection
        """
        
        results = self.collection.get(where={'source': source_file_path})
        present_ids = results['ids']

        return True if len(present_ids) > 0 else False
    
