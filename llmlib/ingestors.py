from llmlib.chunkers import TextChunker
from llmlib.text_readers import PDFTextReader
from llmlib.vectorestore_components import CustomChromaDB

class BaseTextIngestor(object):
    """
    Base class to implement ingestion process of documents.

    The process of ingestion is as follows:
    Raw document -> Convert to text (list of strings) -> chunk the incoming list of string into smaller
    chunks of strings -> convert list of strings into their respective vector embeddings using custom chroma
    embedding function -> store the chroma embeddings into chroma vector database
    """

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 db_path: str,
                 collection_name: str) -> None:
        """
        Method to instantiate object of :class: Ingestor.

        :Parameters:
        model_path: Absolute path to the embedding model.
        tokenizer_path: Absolute path to the tokenizer corresponding to the embedding model.
        db_path: Absolute path to the directory which will act as the vector store.
        collection_name: Name of the collection in which the embedding are to be stored.

        :Returns:
        None
        """

        super().__init__()

        # This chroma collection will create the embedding function on its own
        self._chroma_collection = CustomChromaDB(model_path=model_path,
                                                 tokenizer_path=tokenizer_path,
                                                 db_path=db_path,
                                                 collection_name=collection_name)
        
    def ingest_documents(self) -> None:
        """
        Method that each subclass of :class: Ingestor needs to be implement for its own ingestion implementation
        """

        pass


class PDFTextIngestor(BaseTextIngestor):
    """
    Class to implement ingestion process for PDFs using pymupdf class. This class ingests a single pdf document at a
    time.

    The class ingests a single PDF at a time.
    """

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 db_path: str,
                 collection_name: str,
                 text_reader: PDFTextReader,
                 chunker: TextChunker) -> None:
        """
        Method to instantiate object of :class: PDFIngestor.

        :Parameteres:
        model_path: Absolute path to the embedding model.
        tokenizer_path: Absolute path to the embedding model's tokenizer.
        db_path: Absolute path to the vector store directory.
        collection_name: Name of the collection which will store vector embeddings.
        text_reader: Text reader object to be used for converting raw PDFs containing text.
        chunker: Text chunker to be used for chunking PDF

        :Returns:
        None
        """

        super().__init__(model_path=model_path,
                         tokenizer_path=tokenizer_path,
                         db_path=db_path,
                         collection_name=collection_name)
        
        self._reader = text_reader
        self._chunker = chunker
        self.__collection_name = collection_name

    def ingest_document(self, source_file_path: str) -> bool:
        """
        Method to ingest a PDF document.

        :Parameters:
        source_file_path: Absolute path to the document to be ingested.

        :Returns:
        Boolean representing the completion of the process
        """

        if self._chroma_collection.source_documents_exists(source_file_path=source_file_path):
            print(f"""'{source_file_path}' document already exists in the '{self.__collection_name}' collection, therefore skipping it...""")
        else:
            # Instantiate a list to store all chunks of the text together
            all_text_chunks = list()
            # List of all PDF pages as text List[str]
            text_pages = self._reader.convert_to_text(source_file_path=source_file_path)
            # Loop through each page, generate text chunks and add them to the list
            for page in text_pages:
                text_chunk = self._chunker.chunk(document=page)
                all_text_chunks.extend(text_chunk)

            # Print to the console how many chunks were added to the collection for the given
            # `source_file_path`
            print(f'{len(all_text_chunks)} chunks are being ingested into the collection for the source file: {source_file_path}')

            # Add all chunks related to the file into the collection
            self._chroma_collection.add_documents(documents=all_text_chunks, source_file_path=source_file_path)
        
        return True