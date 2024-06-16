from string import Template


class BasePrompt(object):
    """
    Base class to be inherited by all different types of prompt classes.
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: BasePromptTemplate.

        :Parameters:
        None

        :Returns:
        None
        """

        super().__init__()

    def generate_prompt(self) -> None:
        """
        Method to generate the prompt based on user inputs.
        Each subclass of :class: BasePromptTemplate needs to implement this method on its own

        :Parameters:
        None

        :Returns:
        None
        """

        pass


class AlpacaIRPrompt(BasePrompt):
    """
    Class to produce Alpaca model based 'Instruction-Response' prompt template. 
    """

    def __init__(self, base_context: str | None = None) -> None:
        """
        Method to instantiate object of :class: AlpacaIRTemplate

        :Parameters:
        base_context: Base context that needs to be added to the prompt to provide context
                      about how to deal with user instruction

        :Returns:
        None
        """

        super().__init__()

        
        _DEFAULT_BASE_CONSTEXT = ' '.join(["""Below is an instruction that describes a task.""",
                                           """Write a response that appropriately completes the request."""])
        # set base context
        self._base_context =  base_context if base_context else _DEFAULT_BASE_CONSTEXT

        # prompt template to be used
        self._prompt_template = Template('\n'.join(["""${base_context}""",
                                                    """""",
                                                    """### Instruction:""",
                                                    """${user_instruction}""",
                                                    """""",
                                                    """### Response:"""]))

    def generate_prompt(self, user_instruction: str) -> str:
        """
        Method to generate Alpaca model based prompt using user's instruction.

        :Parameters:
        user_instruction: String representing user's instruction

        :Returns:
        Alpace model base 'Instruction Response' prompt using user's instruction
        """

        return self._prompt_template.safe_substitute({'base_context': self._base_context,
                                                      'user_instruction': user_instruction})


class ChatMLPrompt(BasePrompt):
    """
    Class to produce ChatML based prompt template.
    """

    def __init__(self, system_message: str | None = None) -> None:
        """
        Method to instantiate object of :class: ChatMLTemplate

        :Parameters:
        system_message: Base context that needs to be added to the prompt to provide context
                        about how to deal with user instruction.

        :Returns:
        None
        """

        super().__init__()

        # set base context
        self._system_message =  system_message if system_message else ' '.join(["""You are a chatbot.""",
                                                                                """Provide a factual response to the user's question."""])

        # prompt template to be used
        self._prompt_template = Template("\n".join(["""<|im_start|>system""",
                                                    """${system_message}<|im_end|>""",
                                                    """<|im_start|>user""",
                                                    """${prompt}<|im_end|>""",
                                                    """<|im_start|>assistant"""]))

    def generate_prompt(self, user_instruction: str) -> str:
        """
        Method to generate ChatML based prompt using user's instruction.

        :Parameters:
        user_instruction: String representing user's instruction

        :Returns:
        ChatML based prompt using user's instruction
        """

        return self._prompt_template.safe_substitute({'system_message': self._system_message,
                                                      'prompt': user_instruction})
