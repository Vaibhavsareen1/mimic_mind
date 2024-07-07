from llama_cpp import Llama
from typing import Sequence, List, Tuple


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
                 max_tokens: int,
                 temperature: int,
                 echo: bool = False,
                 n_gpu_layers: int = 0,
                 context_size: int = 0) -> None:
        """
        Method to instantiate object of :class: LlamaCPPGenerator

        :Parameters:
        model_path: Absolute path to the stored GGUF LLM model
        context_size: Size of the model's context.
        max_tokens: Total number of tokens to be generated
        temperature: Value to improve creativity of the output
        echo: Boolean value indicating whether the model should return the user's prompt or not
        n_gpu_layers: Number of model layers to be offloaded to the GPU.

        :Returns:
        None
        """

        super().__init__()


        self._llm_model = Llama(model_path=model_path,
                                n_gpu_layers=n_gpu_layers,
                                n_ctx=context_size)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._echo = echo

    def generate(self, prompt: str) -> str:
        """
        Method to generate output based on the user given prompt using a llamacpp LLM model

        :Parameters:
        prompt: User prompt to be used to generate output

        :Returns:
        output provided by the LLM
        """

        output = self._llm_model.create_completion(prompt=prompt,
                                                   max_tokens=self._max_tokens,
                                                   temperature=self._temperature,
                                                   echo=self._echo)
        
        generated_output = output['choices'][0]['text']

        return generated_output
