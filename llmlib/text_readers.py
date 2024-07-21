import fitz
from typing import List


class BaseTextReader(object):
    """
    Base text reader class for converting documents of any format text.
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: BaseTextReader.
        
        :Parameters:
        None

        :Returns:
        None
        """

        super().__init__()

    def convert_to_text(self) -> None:
        """
        Method that each subclass of :class: BaseTextReader needs to overwrite in order to implement their own version
        of text conversion
        """

        pass


class PDFTextReader(BaseTextReader):
    """
    A class that is used to extract text from a PDF file.
    """

    def __init__(self, single_text: bool = False) -> None:
        """
        Method to instantiate object of :class: PDFTextReader

        :Parameters:
        single_text: Boolean to concatenate content present across multiple pages into a single string

        :Returns:
        None
        """

        super().__init__()

        self._single_text = single_text

    def convert_to_text(self, source_file_path: str, ) -> List[str]:
        """
        Method to convert a pdf file into text. Text from each page will be stored as a separate string.
        All strings are stored and returned as a List of strings

        :Parameters:
        source_file_path: Absolute file path of the pdf file

        :Returns:
        string representation of the pdf file
        """

        # Instantiate a list to store pages of the pdf documents as strings
        all_pages = list()

        # Load PDF file.
        with fitz.open(source_file_path) as doc:
            # Loop through all pages present in the document and store them in `all_pages` list in the form of string
            for page in doc:
                all_pages.append(page.get_text())

        return [" ".join(all_pages)] if self._single_text else all_pages 
    

class TextFileReader(BaseTextReader):
    """
    A class that is used to extract text from a text file.
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: TextFileReader

        :Parameters:
        None

        :Returns:
        None
        """

        super().__init__()

    def convert_to_text(self, source_file_path: str) -> List[str]:
        """
        Method to extract text and store the it in the form of a list of strings from a text file

        :Parameters:
        source_file_path: Absolute file path of the text file

        :Returns:
        string representation of the text file
        """

        with open(source_file_path, mode='r') as doc:
            # Store the entire text file as a single text string
            all_text = doc.read()

        return all_text