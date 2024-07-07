import json
from typing import Dict, List, Tuple


class VectorstoreInfoHandler(object):
    """
    Class to handle information related to the vector databases. This class will store information about existing 
    vector databases such as the retriever model used, chunk size of the documents for the vector database and documents
    that have been ingested and are present in the current vector database.
    """

    def __init__(self, vectorstore_config_path: str) -> None:
        """
        Method to initialize object of :class: VectorstoreInfoHandler.

        :Parameters:
        vectorstore_config_path: Path to the vectore store

        :Returns:
        None
        """

        super().__init__()

        # Place holders for attributes to be used to control which collection name to work with.
        # `_collection_name` and `_collection_info` are attributes to store current collection's name and info
        self._collection_name: str = ''
        self._collection_info: Dict[str, float | int | List | str] = dict()
        self._collections_info: Dict[str, Dict[str, float | int | List | str]] = dict()
        # List to store name of all existing collections
        self._all_collection_list: List[str] = list()
        # List to store ingested documentes for the 'current' collection
        self._ingested_documents: List[str] = list()

        # Raise an exception if the variable `model_config_path` does not satisfy below constraints
        if (not isinstance(vectorstore_config_path, str)):
            raise TypeError(' '.join(['`vectorstore_path` has to of the type `str`.',
                                     f'You have passed a value of type {type(vectorstore_config_path)}']))
        
        if not vectorstore_config_path.endswith('.json'):
            raise ValueError(' '.join(['The model configuration file should be a JSON file.',
                                      f'You have provided {vectorstore_config_path}']))
    
        # Assign vectorstore_config_path to be used within the object
        self._config_file_path = vectorstore_config_path
        # Load configurations for existing collections
        self._generate_collections_info()

        # Set default collection information
        _DEFAULT_COLLECTION = self._all_collection_list[0] if len(self._all_collection_list) > 0 else 'No_Collection'
        self._set_current_collection(collection_name=_DEFAULT_COLLECTION)

        # Generate a List containing names of all available vectore databases
        self._generate_all_collection_list()

    def add_collection(self,
                       collection_name: str,
                       retriever_model_name: str,
                       chunk_size: int,
                       overlap_size: int) -> bool:
        """
        Method to add information about a new collection that is being added to the vectorstore

        :Parameters:
        collection_name: Name of the collection being added
        retriever_model_name: Name of the retriever model used
        chunk_size: Size of chunks used during the ingestion of documents
        overlap_size: The jump between words in order to retain context

        :Returns:
        A boolean representing the completion of the process
        """

        # Step 1: Add it to preloaded collections_info attribute
        # Check if it is a new collection name
        if collection_name in self.get_collection_list():
            raise Exception(" ".join(['The collection name you have provided already exists.',
                                      'Please use a different name or delete the previous collection']))
        
        self._collections_info[collection_name] = {'Retriever_Model_Name': retriever_model_name,
                                                   'Chunk_Size': chunk_size,
                                                   'Overlap_Size': overlap_size,
                                                   'Documents_Ingested': []}
        # Update collection list
        self._generate_all_collection_list()

        # Step 2: Update the original collections info file
        # Load existing vector store configuration file
        with open(self._config_file_path, mode='r', encoding='utf-8') as file:
            file_content = file.read()

        vectorstore_configs: Dict[str, Dict[str, float | int | str]] = json.loads(file_content)

        # If the collection already exists then raise an error
        if collection_name in vectorstore_configs.keys():
            raise Exception('The collection name already exists. Please use a different name or delete the previous collection')
        
        # load information about the new collection into the config file and save it
        vectorstore_configs[collection_name] = {'Retriever_Model_Name': retriever_model_name,
                                                'Chunk_Size': chunk_size,
                                                'Overlap_Size': overlap_size,
                                                'Documents_Ingested': []}

        # Step 3: Save the updated colllection info file in the JSON format
        with open(self._config_file_path, mode='w', encoding='utf-8') as file:
            json.dump(vectorstore_configs, file, indent=4)

        return True

    def add_document(self, document_name: str) -> bool:
        """
        Method to add name of document to the existing collection. This method does not check if the document already
        exist or not. Check in the document list if the document already exists or not.

        :Parameters:
        document_name: Name of the document to be added

        :Returns:
        Boolean representing the completion of the process
        """

        # Check if document already exists
        if document_name in self._ingested_documents:
            raise Exception(' '.join[f'{document_name} has already been ingested.',
                                     'Make sure are not trying to ingest duplicate documents'])
        
        # Add document to the current list of ingested documents
        self._collection_info['Documents_Ingested'].append(document_name)

        # Updated ingested documents list
        self._generate_all_documents_list()

        # Update the vectorstore confiuration file
        # Step 1: Load the vectorstore configuration file
        with open(self._config_file_path, mode='r', encoding='utf-8') as file:
            file_content = file.read()
            vectorstore_config = json.loads(file_content)

        # Check if document already exists
        if document_name in vectorstore_config[self._collection_name]['Documents_Ingested']:
            raise Exception(' '.join[f'{document_name} has already been ingested.',
                                     'Make sure are not trying to ingest duplicate documents'])

        # Step 2: Update the list of documents for the current collection
        vectorstore_config[self._collection_name]['Documents_Ingested'].append(document_name)

        # Step 3: Save the configuration file
        with open(self._config_file_path, mode='w', encoding='utf-8') as file:
            json.dump(vectorstore_config, file, indent=4)
    
        return True

    def delete_collection(self, collection_name: str) -> bool:
        """
        Method to delete information about an existing collection

        :Parameters:
        collection_name: Name of the collection you want to delete

        :Returns:
        """

        if collection_name not in self._all_collection_list:
            raise Exception(" ".join(['The collection you are trying to delete does not exist.',
                                      'Please provide a name which exists.',
                                      f'You have provide {collection_name} as the value for `collection_name`']))
        
        # Load stored collections information
        with open(self._config_file_path, mode='r', encoding='utf-8') as input_file:
            file_content = input_file.read()

        vectorestore_configs: Dict[str, Dict[str, float | int | str]] = json.loads(file_content)

        # Temporary dict to be used for updating config file before storing them
        temp_storage_dict = dict()
        for name, info in vectorestore_configs.items():
            if collection_name == name:
                continue
            
            temp_storage_dict[name] = info
        
        # Save the updated vector store configuration file
        with open(self._config_file_path, mode='w', encoding='utf-8') as output_file:
            json.dump(temp_storage_dict, output_file, indent=4)

        self._generate_collections_info()
        self._generate_all_collection_list()

        # If the collection that is to be deleted the current collection then update 
        if collection_name == self._collection_name:
            _DEFAULT_COLLECTION = self._all_collection_list[0] if len(self._all_collection_list) > 0 else 'No_Collection'
            self.switch_collection(collection_name=_DEFAULT_COLLECTION)
        
        return True

    def get_collection_list(self) -> List[str]:
        """
        Method to get all the names of collections available in the vectorstore

        :Parameters:
        None

        :Returns:
        List of names of all collections available
        """

        return self._all_collection_list

    def get_collection_name(self) -> str:
        """
        Method to get the name of current loaded collection

        :Parameteres:
        None

        :Returns:
        Name of the current collection
        """

        return self._collection_name

    def get_chunk_size(self) -> int:
        """
        Method to return chunk size for chunking documents,

        :Parameters:
        None

        :Returns:
        Chunk Size to be used with the current collection
        """

        return self._collection_info.get('Chunk_Size', 0)
    
    def get_documents(self) -> List[str]:
        """
        Method to get names of all documents that have been added/ingested in the current collection

        :Parameters:
        None

        :Returns:
        A list of document names that have been ingested within the current collection
        """

        return self._ingested_documents

    def get_overlap_size(self) -> int:
        """
        Method to get the overlap size set for the current collection.

        :Parameters:
        None

        :Returns:
        Returns the overlap size that is used with the current collection
        """

        return self._collection_info.get('Overlap_Size', 0)

    def get_retriever_model(self) -> str:
        """
        Method to get the name of the retriever model which is setup for the current collection.

        :Returns:
        Name of the retriever model associated with the current collection.
        If no retriever is used then defaults to value None which needs to be handled separately
        """

        return self._collection_info.get('Retriever_Model_Name', 'No Collection Exists')

    def switch_collection(self, collection_name: str) -> bool:
        """
        Method to switch current collection to the new collection requested by the user. The new
        collection name already exists

        :Parameters:
        collection_name: Name of the collection to be switched with the current collection

        :Returns:
        Boolean representing the completion of the switching process
        """
        
        self._set_current_collection(collection_name=collection_name)

        return True

    def _generate_collections_info(self) -> bool:
        """
        Method to get information related to existing collections from the saved json format

        :Parameters:
        None

        :Returns:
        A boolean representing completion of the process
        """

        # Load model information file
        with open(self._config_file_path, mode='r', encoding='utf-8') as input_file:
            file_content = input_file.read()
            self._collections_info = json.loads(file_content)

        return True
    
    def _generate_all_collection_list(self) -> bool:
        """
        Method to generate a list containing names of all existing vector databases in the current vector store

        :Parameters:
        None

        :Returns:
        A boolean representing the completion of the process
        """

        if len(list(self._collections_info.keys())) == 0:
            self._all_collection_list = ['No_Collection']
        else:
            self._all_collection_list = sorted([name for name in self._collections_info.keys()],
                                             reverse=False)
            
        return True

    def _generate_all_documents_list(self) -> bool:
        """
        Method to generate a list containing names of all documents that been ingested in the current document
        collection.

        :Parameters:
        None

        :Returns:
        A boolean value to represent completion of the process.
        """
 
        # Store documents in sorted order if documents have been ingested otherwise 
        if len(self._collection_info['Documents_Ingested']) == 0:
            self._ingested_documents = ['No documents exists']
        else:
            self._ingested_documents = sorted(self._collection_info['Documents_Ingested'],
                                              reverse=False)
            
        return True

    def _set_current_collection(self, collection_name: str) -> bool:
        """
        Method to set configure current collection's information

        :Parameters:
        collection_name: Name of the collection that will become the new 'current' collection

        :Returns:
        Boolean representing completion of the process
        """

        self._collection_name = collection_name
        if self._collection_name == 'No_Collection':
            self._collection_info = {'Retriever_Model_Name': 'No Collection Exists',
                                     'Chunk_Size': 0,
                                     'Overlap_Size': 0,
                                     'Documents_Ingested': []}
        else:
            self._collection_info = self._collections_info.get(self._collection_name)

        # Update ingested documents list
        self._generate_all_documents_list()

        return True
