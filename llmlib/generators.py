import time
from llama_cpp import Llama
from typing import Sequence, List, Tuple
from openai import OpenAI
from openai import (APIError, APIConnectionError, APITimeoutError, InternalServerError)


class BaseGenerator(object):
    """
    Base class which should be inherited by all subclasses to implement token
    generation using LLMs.
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: Generator.

        :Parameters:
        None

        :Returns:
        None
        """

        super().__init__()

    def generate(self) -> None:
        """
        Method each and every subclass of :class: Generator needs to implement
        for generation process using LLMs

        :Parameters:
        None
        
        :Returns
        """

        pass


class LlamaCPPGenerator(BaseGenerator):
    """
    Class that generates token sequences using llama-cpp bindings where models are in the GGUF format
    """

    def __init__(self,
                 model_path: str,
                 context_size: int = 0,
                 echo: bool = False,
                 max_tokens: int = 2048,
                 n_gpu_layers: int = 0,
                 retries: int = 5,
                 temperature: int | float = 0,
                 wait_time: int = 5) -> None:
        """
        Method to instantiate object of :class: LlamaCPPGenerator

        :Parameters:
        model_path: Absolute path to the stored GGUF LLM model.
        echo: Boolean value indicating whether the model should return the user's prompt or not
        context_size: Size of the model's context.
        max_tokens: Total number of tokens to be generated.
        n_gpu_layers: Number of model layers to be offloaded to the GPU.
        retries: Number of retries to perform before throwing an error.
        temperature: The temperature value to induce creativity within the model. The temperature is set to 0 as 
                    default as we want the model to be as deterministic as possible.
        wait_time: The amount of time the model needs to wait before it retries sending the request to the model.

        :Returns:
        An instance of :class: LlamaCPPGenerator
        """

        super().__init__()
        # Static attributes
        self._client = Llama(model_path=model_path,
                             n_gpu_layers=n_gpu_layers,
                             n_ctx=context_size)
        self._echo: bool = echo
        self._max_tokens: int = max_tokens
        self._retries: int = retries
        self._temperature: float = temperature
        self._wait_time: int = wait_time

        # Dynamic attributes
        self._choices: List[str] = []
        self._finish_reasons: List[str] = []
        self._completion_tokens: int = 0        
        self._prompt_tokens: int = 0
        self._total_tokens: int = 0

    def generate(self, user_message: str | None = None) -> List[str]:
        """
        Method to generate output based on the user given prompt using a llamacpp LLM model

        :Parameters:
        user_message: User prompt to be used to generate output

        :Returns:
        output provided by the LLM
        """

        # Raise an exception if no message is provided by the user
        if user_message is None or user_message == '':
            raise Exception(f'Please provide a valid message. You have provide "{user_message}"')

        self._get_response_info(user_message=user_message)
        
        return self._choices

    def get_finish_reasons(self) -> List[str]:
        """
        Method to get a list of reasons for all choices for which the model had stopped generating tokens.

        :Parameters:
        None

        :Returns:
        List of reasons
        """

        self._finish_reasons

    def get_completion_tokens(self) -> int:
        """
        Method to get how many tokesn were used in the completion phase.

        :Parameters:
        None

        :Returns:
        total token count
        """

        return self._completion_tokens

    def get_prompt_tokens(self) -> int:
        """
        Method to get how many tokesn were used in the prompt phase.

        :Parameters:
        None

        :Returns:
        total token count
        """

        return self._prompt_tokens

    def get_total_tokens(self) -> int:
        """
        Method to get how many tokesn were used in the prompt and completion phase.

        :Parameters:
        None

        :Returns:
        total token count
        """

        return self._total_tokens
    
    def _get_response_info(self, user_message: str) -> True:
        """
        Method to get a response from the Open-AI model based on use's input and store necessary information related
        to the response.

        :Parameters:
        user_message: Message provide by the user

        :Returns:
        Boolean representing end of the process
        """

        IS_SUCCESSFUL = False
        RETRY_COUNTER = 0

        # Reset dynamic attributes
        self._reset_attributes()

        while not IS_SUCCESSFUL:
            try:
                response = self._client.create_completion(prompt=user_message,
                                                          max_tokens=self._max_tokens,
                                                          temperature=self._temperature,
                                                          echo=self._echo)
                
                # Store neccessary information from the response if it is successful
                self._choices = [choice['text'] for choice in response['choices']]
                self._finish_reasons = [choice['finish_reason'] for choice in response['choices']]
                self._completion_tokens = response['usage']['completion_tokens']
                self._prompt_tokens = response['usage']['prompt_tokens']
                self._total_tokens = response['usage']['total_tokens']

                # The request went through there fore Update IS_SUCCESSFUL to break out of the while loop
                IS_SUCCESSFUL = True
            
            except Exception:
                # Jump out of the loop if the model exceeds the number of retries user wants the model to try
                if RETRY_COUNTER >= self._retries:
                    IS_SUCCESSFUL = True

                print(f"There was an issue with llama cpp package. Retrying in {self._wait_time} seconds.")
                time.sleep(self._wait_time)

        return True
    
    def _reset_attributes(self) -> bool:
        """
        Method to reset dynamic attributes which are displayed to the user.

        :Parameters:
        None

        :Returns:
        Boolean representing completion of the process
        """

        self._completion_tokens = 0
        self._prompt_tokens = 0
        self._total_tokens = 0
        self._choices.clear()
        self._finish_reasons.clear()

        return True


class OpenAISingleTurnGenerator(BaseGenerator):
    """
    Class that generates token sequences using models from open AI family. This class deals with simple single
    turn dialogue conversation. 

    Limitations:
    Only text sequences can be generated.
    Does not support function calling.
    """

    def __init__(self,
                 api_key: str,
                 model_name: str,
                 max_tokens: int | None = None,
                 number_of_responses: int = 1,
                 retries: int = 5,
                 system_message: str | None = None,
                 temperature: int | float = 0,
                 wait_time: int = 5) -> None:
        """
        Method to instantiate object of :class: OpenAIGenerator
        
        :Parameters:
        api_key: API Key to be used during the generation phase.
        model_name: Name of the model to be used for generation of tokens.
        max_tokens: Maximum length of the response provided by the model.
        number_of_responses: Number of responses the model has to generate.
        retries: Number of retries to perform before throwing an error.
        system_message: System message to be used during the session.
        temperature: The temperature value to induce creativity within the model. The temperature is set to 0 as 
                    default as we want the model to be as deterministic as possible.
        wait_time: The amount of time the model needs to wait before it retries sending the request to the Open-AI
                   model.

        :Returns:
        An instance of :class: OpenAISingleTurnGenerator
        """

        super().__init__()
        # Static attributes
        self._client = OpenAI(api_key=api_key)
        self._model_name: str = model_name
        self._max_tokens: int = max_tokens
        self._number_of_responses: int = number_of_responses
        self._retries: int = retries
        self._system_message: str = system_message if system_message else 'You are a helpful assistant.'
        self._temperature: float = temperature
        self._wait_time: int = wait_time

        # Dynamic attributes
        self._choices: List[str] = []
        self._finish_reasons: List[str] = []
        self._completion_tokens: int = 0        
        self._prompt_tokens: int = 0
        self._total_tokens: int = 0

    def generate(self, user_message: str | None = None) -> List[str]:
        """
        Method to get a response from the Open-AI model based on user's message. The model can produce
        `_number_of_responses` responses.

        :Parameters:
        user_message: Message from the user

        :Returns:
        List of response strings produced by the model
        """

        # Raise an exception if no message is provided by the user
        if user_message is None or user_message == '':
            raise Exception(f'Please provide a valid message. You have provide "{user_message}"')

        self._get_response_info(user_message=user_message)
        
        return self._choices

    def get_finish_reasons(self) -> List[str]:
        """
        Method to get a list of reasons for all choices for which the model had stopped generating tokens.

        :Parameters:
        None

        :Returns:
        List of reasons
        """

        self._finish_reasons

    def get_completion_tokens(self) -> int:
        """
        Method to get how many tokesn were used in the completion phase.

        :Parameters:
        None

        :Returns:
        total token count
        """

        return self._completion_tokens

    def get_prompt_tokens(self) -> int:
        """
        Method to get how many tokesn were used in the prompt phase.

        :Parameters:
        None

        :Returns:
        total token count
        """

        return self._prompt_tokens

    def get_total_tokens(self) -> int:
        """
        Method to get how many tokesn were used in the prompt and completion phase.

        :Parameters:
        None

        :Returns:
        total token count
        """

        return self._total_tokens
    
    def _get_response_info(self, user_message: str) -> True:
        """
        Method to get a response from the Open-AI model based on use's input and store necessary information related
        to the response.

        :Parameters:
        user_message: Message provide by the user

        :Returns:
        Boolean representing end of the process
        """

        IS_SUCCESSFUL = False
        RETRY_COUNTER = 0

        # Reset dynamic attributes
        self._reset_attributes()

        while not IS_SUCCESSFUL:
            try:
                response = self._client.chat.completions.create(
                                model=self._model_name,
                                messages=[{"role": "system", "content": self._system_message},
                                          {"role": "user", "content": user_message}],
                                max_tokens=self._max_tokens,
                                temperature=self._temperature,
                                n=self._number_of_responses)
                
                # Store neccessary information from the response if it is successful
                self._choices = [choice.message.content for choice in response.choices]
                self._finish_reasons = [choice.finish_reason for choice in response.choices]
                self._completion_tokens = response.usage.completion_tokens
                self._prompt_tokens = response.usage.prompt_tokens
                self._total_tokens = response.usage.total_tokens

                # The request went through there fore Update IS_SUCCESSFUL to break out of the while loop
                IS_SUCCESSFUL = True
            
            except (APIError, APIConnectionError, InternalServerError, APITimeoutError):
                # Jump out of the loop if the model exceeds the number of retries user wants the model to try
                if RETRY_COUNTER >= self._retries:
                    IS_SUCCESSFUL = True

                print(f"There was an issue on Open-AI's side while processing the request. Retrying in {self._wait_time} seconds.")
                time.sleep(self._wait_time)

        return True
    
    def _reset_attributes(self) -> bool:
        """
        Method to reset dynamic attributes which are displayed to the user.

        :Parameters:
        None

        :Returns:
        Boolean representing completion of the process
        """

        self._completion_tokens = 0
        self._prompt_tokens = 0
        self._total_tokens = 0
        self._choices.clear()
        self._finish_reasons.clear()

        return True