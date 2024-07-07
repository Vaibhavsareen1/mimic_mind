import json
from typing import Dict, List, Tuple


class GenModelInfoHandler(object):
    """
    Class that handles information about LLM models used for the 'generator' process. This class handles logic to
    switch models based on user preference. The information regarding the models should be stored in a JSON format.
    """

    def __init__(self, model_config_path: str) -> None:
        """
        Method to instantiate object of :class: ModelInfoHandler

        :Parameters:
        model_config_path: Path to file containing information related to LLM models

        :Returns: None
        """

        super().__init__()
        
        # place holders for attributes to be used within the model
        self._model_name: str = ''
        self._model_config: Dict[str, float | int | str] = dict()
        self._configurations_info: Dict[str, Dict[str, float | int | str]] = dict()
        self._all_model_list: List[str] = list()

        # Raise an exception if the variable `model_config_path` does not satisfy below constraints
        if (not isinstance(model_config_path, str)):
            raise TypeError(' '.join(['`model_config_path` has to of the type `str`.',
                                     f'You have passed a value of type {type(model_config_path)}']))
        
        if not model_config_path.endswith('.json'):
            raise ValueError(' '.join(['The model configuration file should be a JSON file.',
                                      f'You have provided {model_config_path}']))
        
        # Assign model_config_path to be used within the object
        self._config_file_path = model_config_path

        # Load file containing all model configurations
        self._get_configurations_info()
        # Generate a iist containing names of the available models
        self._generate_all_model_list()

        # Default model name to be set as the current model
        _DEFAULT_MODEL = self._all_model_list[0]

        # Set default current model
        self._set_current_model(model_name=_DEFAULT_MODEL)

    def get_api_key(self) -> str | None:
        """
        Method to get the API key associated with the model if it exists.

        :Parameters:
        None

        :Returns:
        API key to use the model
        """

        return self._model_config.get('API_Key_variable')
    
    def get_api_secret(self) -> str | None:
        """
        Method to get the api secret associated with the model if it exists.

        :Parameters:
        None

        :Returns:
        API Secret to use the model
        """

        return self._model_config.get('API_Secret_variable')

    def get_context_length(self) -> int:
        """
        Method to get the context length of the current model

        :Parameters:
        None

        :Returns:
        Context length of the current model
        """

        return self._model_config.get('Context_Length')

    def get_foundation_model_name(self) -> str | None:
        """
        Method to get the name of the foundation model associated with the model if it exists.

        :Parameters:
        None

        :Returns:
        Name of the foundation model
        """
        
        return self._model_config.get('Foundation_model_name')

    def get_model_config(self) -> Dict[str, float | int | str]:
        """
        Method to get the configuration information of the current model

        :Parameters:
        None

        :Returns:
        Dictionary containing configuration information about the current model
        """

        return self._model_config

    def get_model_list(self) -> List[str]:
        """
        Method to get the list of names of all existing models

        :Parameters:
        None

        :Returns:
        List of names of all existing models
        """

        return self._all_model_list

    def get_model_name(self) -> str:
        """
        Method to get the name of the current model in use.

        :Parameters:
        None

        :Returns:
        Name of the current model
        """

        return self._model_name

    def get_model_path(self) -> str:
        """
        Method to get the path to the current model

        :Parameters:
        None

        :Returns:
        Path to the current model
        """

        return self._model_config.get('Model_Path')
    
    def get_model_type(self) -> str:
        """
        Method to get what category the current model belongs to. The model either can be an open sourced model
        or a foundation private model.

        :Parameters:
        None

        :Returns:
        Category the model belongs to.
        """

        return self._model_config.get('Model_Type')
    
    def switch_model(self, model_name: str) -> bool:
        """
        Method to switch the current model with the model required by the user

        :Parameters:
        model_name: Name of the model which should replace the current model

        :Return:
        Boolean value representing completion of the process
        """
        
        self._set_current_model(model_name=model_name)

        return True

    def _get_configurations_info(self) -> bool:
        """
        Method to load all model and their configurations available in the configuration file
 
        :Parameters:
        None

        :Returns:
        Boolean value to represent completion of the process
        """

        # Load model information file
        with open(self._config_file_path, mode='r', encoding='utf-8') as input_file:
            file_content = input_file.read()
            model_configs = json.loads(file_content)

        # Raise an exception if the file does not contain any model configurations in it
        if len(model_configs) < 1:
            raise Exception(''.join(['There is no model configuration available in the current file. ',
                                     'Please provide a file that contains model configurations']))
        
        self._configurations_info = model_configs

        return True

    def _generate_all_model_list(self) -> bool:
        """
        Method to generate a list of all models available in the configuration file provided by the user.

        :Parameters:
        None

        :Returns:
        A boolean to represent the completion of the process
        """
        
        self._all_model_list = sorted([model_name for model_name in self._configurations_info.keys()],
                                      reverse=False)

        return True

    def _set_current_model(self, model_name: str) -> bool:
        """
        Method to set current model and its configuration to be used by the user

        :Parameters:
        model_name: Model to be set as the 'current' model to be used by the user

        :Returns:
        Boolean representing the completion of the process
        """

        self._model_name = model_name
        self._model_config = self._configurations_info.get(self._model_name)

        return True


class RetModelInfoHandler(object):
    """
    Class that handles information about LLM models used for the 'retrieval' process. This class handles logic to
    switch models based on user preference. The information regarding the models should be stored in a JSON format.
    """

    def __init__(self, model_config_path: str) -> None:
        """
        Method to instantiate object of :class: ModelInfoHandler

        :Parameters:
        model_config_path: Path to file containing information related to LLM models

        :Returns: None
        """

        super().__init__()
        
        # place holders for attributes to be used within the model
        self._model_name: str = ''
        self._model_config: Dict[str, float | int | str] = dict()
        self._configurations_info: Dict[str, Dict[str, float | int | str]] = dict()
        self._all_model_list: List[str] = list()

        # Raise an exception if the variable `model_config_path` does not satisfy below constraints
        if (not isinstance(model_config_path, str)):
            raise TypeError(''.join(['`config_path` has to of the type `str`. ',
                                     f' You have passed values of type {type(model_config_path)}']))
        
        if not model_config_path.endswith('.json'):
            raise ValueError(''.join(['The model configuration file should be a JSON file. ',
                                      f' You have provided {model_config_path}']))
        
        # Assign model_config_path to be used within the object
        self._config_file_path = model_config_path

        # Load file containing all model configurations
        self._get_configurations_info(config_path=self._config_file_path)
        # Generate a iist containing names of the available models
        self._generate_all_model_list()

        # Default model name to be set as the current model
        _DEFAULT_MODEL = self._all_model_list[0]

        # Set default current model
        self._set_current_model(model_name=_DEFAULT_MODEL)

    def get_api_key(self) -> str | None:
        """
        Method to get the API key associated with the model if it exists.

        :Parameters:
        None

        :Returns:
        API key to use the model
        """

        return self._model_config.get('API_Key_variable')
    
    def get_api_secret(self) -> str | None:
        """
        Method to get the api secret associated with the model if it exists.

        :Parameters:
        None

        :Returns:
        API Secret to use the model
        """

        return self._model_config.get('API_Secret_variable')

    def get_embedding_dimension(self) -> int:
        """
        Method to get the context length of the current model

        :Parameters:
        None

        :Returns:
        Context length of the current model
        """

        return self._model_config.get('Embedding_Dimension')

    def get_foundation_model_name(self) -> str | None:
        """
        Method to get the name of the foundation model associated with the model if it exists.

        :Parameters:
        None

        :Returns:
        Name of the foundation model
        """
        
        return self._model_config.get('Foundation_model_name')

    def get_max_tokens(self) -> int:
        """
        Method to get maximum number of tokens the retrieval model can take to generate embeddings

        :Parameters:
        None

        :Returns:
        Total number of tokens a model can take
        """

        return self._model_config.get('Max_Tokens')

    def get_model_config(self) -> Dict[str, float | int | str]:
        """
        Method to get the configuration information of the current model

        :Parameters:
        None

        :Returns:
        Dictionary containing configuration information about the current model
        """

        return self._model_config

    def get_model_list(self) -> List[str]:
        """
        Method to get the list of names of all existing models

        :Parameters:
        None

        :Returns:
        List of names of all existing models
        """

        return self._all_model_list

    def get_model_name(self) -> str:
        """
        Method to get the name of the current model in use.

        :Parameters:
        None

        :Returns:
        Name of the current model
        """

        return self._model_name

    def get_model_path(self) -> str:
        """
        Method to get the path to the current model

        :Parameters:
        None

        :Returns:
        Path to the current model
        """

        return self._model_config.get('Model_Path')

    def get_model_size(self) -> str:
        """
        Method to get the size of the retrieval LLM model.

        :Parameters:
        None

        :Returns:
        Size of the retrieval model
        """

        return self._model_config.get('Model_Size')

    def get_model_type(self) -> str:
        """
        Method to get what category the current model belongs to. The model either can be an open sourced model
        or a foundation private model.

        :Parameters:
        None

        :Returns:
        Category the model belongs to.
        """

        return self._model_config.get('Model_Type')

    def get_tokenizer_path(self) -> str:
        """
        Method to get path to the current model's tokenizer.

        :Parameters:
        None

        :Returns:
        Path of the retriever model's tokenizer
        """

        return self._model_config.get('Tokenizer_Path')

    def switch_model(self, model_name: str) -> bool:
        """
        Method to switch the current model with the model required by the user

        :Parameters:
        model_name: Name of the model which should replace the current model

        :Return:
        Boolean value representing completion of the process
        """
        
        self._set_current_model(model_name=model_name)

        return True

    def _get_configurations_info(self, config_path: str) -> bool:
        """
        Method to load all model and their configurations available in the configuration file
 
        :Parameters:
        config_path: Path to the file that stores all model configurations

        :Returns:
        Boolean value to represent completion of the process
        """

        import json
        # Load model information file
        with open(config_path, mode='r', encoding='utf-8') as input_file:
            model_configs = json.load(input_file)

        # Raise an exception if the file does not contain any model configurations in it
        if len(model_configs) < 1:
            raise Exception(''.join(['There is no model configuration available in the current file. ',
                                     'Please provide a file that contains model configurations']))
        
        self._configurations_info = model_configs

        return True

    def _generate_all_model_list(self) -> bool:
        """
        Method to generate a list of all models available in the configuration file provided by the user.

        :Parameters:
        None

        :Returns:
        A boolean to represent the completion of the process
        """
        
        self._all_model_list = sorted([model_name for model_name in self._configurations_info.keys()],
                                      reverse=False)

        return True

    def _set_current_model(self, model_name: str) -> bool:
        """
        Method to set current model and its configuration to be used by the user

        :Parameters:
        model_name: Model to be set as the 'current' model to be used by the user

        :Returns:
        Boolean representing the completion of the process
        """

        self._model_name = model_name
        self._model_config = self._configurations_info.get(self._model_name)

        return True