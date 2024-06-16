from typing import List


class BaseChunker(object):
    """
    Base class to implement the process of chunking of any form of data
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: Chunker
        """
        super().__init__()
    
    def chunk(self) -> None:
        """
        Method that each subclass of :class: BaseChunker needs to overwrite in order to implement its own version of
        chunking.
        """

        pass


class TextChunker(BaseChunker):
    """
    Class to implement chunking of text into smaller chunks based on user defined chunk size and overlap size.
    Chunk size is the size of each chunked text and overlap size is the interval size at which chunking will take
    place in order to retain context of the text.
    """

    def __init__(self, chunk_size: int, overlap_size: int) -> None:
        """
        Method to instantiate object of :class: TextChunker.

        Overlap size should be greater than 0.

        :Parameters:
        chunk_size: size of each document chunk.
        overlap_size: size of overlap between chunks in order to retain context.

        :Returns:
        None
        """

        super().__init__()

        # Store chunk and overlap size provided by the user
        self._chunk_size = chunk_size
        self._overlap_size = overlap_size if overlap_size > 0 else chunk_size

    def chunk(self, document: str) -> List[str]:
        """
        Method to chunk text into smaller chunks of text.

        The document is chunked iteratively. The document is chunked at regular intervals of 'overlap_size' and the
        chunks are of the size 'chunk_size'. After the end of the process we are left with a list of smaller chunks of
        text.

        :Parameters:
        document: Document to be chunked

        :Returns:
        List of smaller chunks of text
        """

        document_chunks = list()

        # Loop through the document and chunk it at regular intervals of 'overlap_size'
        for start_index in range(0, len(document) + 1, self._overlap_size):
            # Add chunks to the list 'document_chunks' of size 'chunk_size'
            document_chunks.append(document[start_index: start_index + self._chunk_size])

        return document_chunks
    