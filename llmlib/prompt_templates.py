from typing import List
from string import Template

class BasePromptTemplate(object):
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
        Each subclass of :class: BasePromptTemplate needs to implement this method

        :Parameters:
        None

        :Returns:
        None
        """

        pass


class Llama2PromptTemplate(BasePromptTemplate):
    """
    Class to produce single turn prompts for LLama-2 models. This prompt class is suitable for for 7B, 13B and 70B
    parameter model.
    """

    def __init__(self, system_message: str | None = None) -> None:
        """
        Method to instantiate object of :class: Llama2PromptTemplate. Pure Llama-2 models don't have any
        specific prompts to use therefore a default prompt is created for it. The default prompt is the same as
        Llama-2-chat prompt.

        :Parameters:
        system_message: Part of the prompt responsible for personification of the model to perform according
                        to user's input.

        :Returns:
        None
        """

        super().__init__()

        _DEFAULT_SYSTEM_MESSAGE = ' '.join(["""You are a helpful, respectful and honest assistant.""",
                                            """Always answer as helpfully as possible, while being safe.""",
                                            """Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.""",
                                            """Please ensure that your responses are socially unbiased and positive in nature.""",
                                            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.""",
                                            """If you don't know the answer to a question, please don't share false information."""])

        # set base context
        self._system_message =  system_message if system_message else _DEFAULT_SYSTEM_MESSAGE

        # prompt template to be used
        self._prompt_template = Template('\n'.join(["""[INST] <<SYS>>""",
                                                    """${system_message}""",
                                                    """<</SYS>>""",
                                                    """${user_message}[/INST]"""]))

    def generate_prompt(self, user_message: str) -> str:
        """
        Method to generate Llama-2 model prompt using user's instruction.

        :Parameters:
        user_message: String representing of user's instruction

        :Returns:
        LLama-2 model's prompt based on user's instruction  
        """

        # If user provides an empty string raise an error
        if (user_message is None) or (user_message == ''):
            raise Exception(f'Please provide an input. You have provided {user_message}')

        return self._prompt_template.safe_substitute({'system_message': self._system_message,
                                                      'user_message': user_message})


class Llama2InstructPromptTemplate(BasePromptTemplate):
    """
    Class to produce single turn prompts for LLama-2-Instruct models. This prompt class is suitable for for 7B, 13B and
    70B parameter model.
    """

    def __init__(self, system_message: str | None = None) -> None:
        """
        Method to instantiate object of :class: Llama2InstructPromptTemplate.

        :Parameters:
        system_message: Part of the prompt responsible for personification of the model to perform according
                        to user's input.

        :Returns:
        None
        """

        super().__init__()

        _DEFAULT_SYSTEM_MESSAGE = ' '.join(["""You are a helpful, respectful and honest assistant.""",
                                            """Always answer as helpfully as possible, while being safe.""",
                                            """Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.""",
                                            """Please ensure that your responses are socially unbiased and positive in nature.""",
                                            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.""",
                                            """If you don't know the answer to a question, please don't share false information."""])

        # set base context
        self._system_message =  system_message if system_message else _DEFAULT_SYSTEM_MESSAGE

        # prompt template to be used
        self._prompt_template = Template('\n'.join(["""[INST] <<SYS>>""",
                                                    """${system_message}""",
                                                    """<</SYS>>""",
                                                    """${user_message}[/INST]"""]))

    def generate_prompt(self, user_message: str) -> str:
        """
        Method to generate Llama-2-Instruct model prompt using user's instruction.

        :Parameters:
        user_message: String representing of user's instruction

        :Returns:
        LLama-2-Instruct model's prompt based on user's instruction  
        """

        # If user provides an empty string raise an error
        if (user_message is None) or (user_message == ''):
            raise Exception(f'Please provide an input. You have provided {user_message}')

        return self._prompt_template.safe_substitute({'system_message': self._system_message,
                                                      'user_message': user_message})


class CodeLlama2InstructPromptTemplate(BasePromptTemplate):
    """
    Class to produce single turn prompts for CodeLLama-2-Instruct models. This prompt class is suitable for for 7B, 13B
    and 70B parameter model
    """

    def __init__(self, system_message: str | None = None) -> None:
        """
        Method to instantiate object of :class: CodeLlama2InstructPromptTemplate

        :Parameters:
        system_message: Part of the prompt responsible for personification of the model to perform according
                        to user's input.

        :Returns:
        None
        """

        super().__init__()

        _DEFAULT_SYSTEM_MESSAGE = ' '.join(["""Write code to solve the following coding problem that obeys the constraints and passes the example test cases.""",
                                            """Please wrap your code answer using ```:"""])
        
        self._system_message = system_message if system_message else _DEFAULT_SYSTEM_MESSAGE

        # prompt template to be used
        self._prompt_template = Template("\n".join(["""[INST] ${system_message}""",
                                                    """${user_message}""",
                                                    """[/INST]"""]))

    def generate_prompt(self, user_message: str) -> str:
        """
        Method to generate CodeLLama-2-Instruct based prompt using user's instruction.

        :Parameters:
        user_message: String representing of user's instruction

        :Returns:
        CodeLlama-2-Instruct model's prompt based on user's instruction 
        """

        # If user provides an empty string raise an error
        if (user_message is None) or (user_message == ''):
            raise Exception(f'Please provide an input. You have provided {user_message}')

        return self._prompt_template.safe_substitute({'system_message': self._system_message,
                                                      'user_message': user_message})


class Llama3PromptTemplate(BasePromptTemplate):
    """
    Class to produce single turn prompt for Llama-3 models. This prompt class is suitable for 8B and 70B parameter
    models. This class can only produce prompts for a single turn conversation. This model does not have a standardized
    format for its prompt which includes system instruction there for creating one to be used with llama-3 models.
    """

    def __init__(self, system_message: str | None = None) -> None:
        """
        Method to instantiate object of :class: Llama3SingleTurnPromptTemplate
        
        :Parameters:
        system_message: Part of the prompt responsible for personification of the model to perform according
                        to user's input.

        :Returns:
        None
        """

        super().__init__()

        _DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant"""

        self._system_message = system_message if system_message else _DEFAULT_SYSTEM_MESSAGE

        # prompt template to be used
        self._prompt_template = Template(" ".join(["""<|begin_of_text|>${system_message}""",
                                                    """${user_message}"""]))

    def generate_prompt(self, user_message: str) -> None:
        """
        Method to generate LLama-3 based prompt using user's instruction.

        :Parameters:
        user_message: String representing of user's instruction

        :Returns:
        Llama-3 model's prompt based on user's instruction 
        """

        # If user provides an empty string raise an error
        if (user_message is None) or (user_message == ''):
            raise Exception(f'Please provide an input. You have provided {user_message}')

        return self._prompt_template.safe_substitute({'system_message': self._system_message,
                                                      'user_message': user_message})


class Llama3InstructSTPromptTemplate(BasePromptTemplate):
    """
    Class to produce single turn prompt for Llama-3-Instruct models. This prompt class is suitable for 8B and 70B 
    parameter models. This class can only produce prompts for a single turn conversation.
    """

    def __init__(self, system_message: str | None) -> None:
        """
        Method to instantiate object of :class: Llama3InstructPromptTemplate.

        :Parameters:
        system_message: Part of the prompt responsible for personification of the model to perform according
                        to user's input.

        :Returns:
        None
        """

        super().__init__()

        _DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant"""

        self._system_message = system_message if system_message else _DEFAULT_SYSTEM_MESSAGE

        # prompt template to be used
        self._prompt_template = Template("\n".join(["""<|begin_of_text|><|start_header_id|>system<|end_header_id|>""",
                                                    """""",
                                                    """${system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>""",
                                                    """""",
                                                    """${user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                                                    ]))

    def generate_prompt(self, user_message: str) -> None:
        """
        Method to generate LLama-3-Instruct based prompt using user's instruction.

        :Parameters:
        user_message: String representing of user's instruction

        :Returns:
        Llama-3 model's prompt based on user's instruction 
        """

        # If user provides an empty string raise an error
        if (user_message is None) or (user_message == ''):
            raise Exception(f'Please provide an input. You have provided {user_message}')

        return self._prompt_template.safe_substitute({'system_message': self._system_message,
                                                      'user_message': user_message})
    

class Llama3InstructMTPromptTemplate(BasePromptTemplate):
    """
    Class to produce multi turn prompt for Llama-3-Instruct models. This prompt class is suitable for 8B and 70B 
    parameter model. This class can produce prompts for Multi-Turn conversations. This class implements simple
    multi-turn prompt and does not care of handling memory generation to reduce the lenght of the prompt. There is
    no limit on the prompt length.
    """

    def __init__(self, system_message: str | None = None) -> None:
        """
        Method to instantiate object of :class: Llama3InstructMTPromptTemplate.

        :Parameters:
        system_message: Part of the prompt responsible for personification of the model to perform according
                        to user's input.

        :Returns:
        None
        """

        super().__init__()

        _DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant"""

        self._system_message = system_message if system_message else _DEFAULT_SYSTEM_MESSAGE

        self._prompt_template = Template("\n".join(["""<|begin_of_text|><|start_header_id|>system<|end_header_id|>""",
                                                    """""",
                                                    """${system_message}<|eot_id|>""",
                                                    """<|start_header_id|>user<|end_header_id|>""",
                                                    """""",
                                                    """${chat_history}"""]))
        
        # Attribute of the class which will store the content of the conversations between the user and the language model 
        self._chat_history: List[str] = list()

    def generate_prompt(self, user_message: str, assistant_message: str | None = None) -> None:
        """
        Method to generate Llama-3-Instruct multi turn prompt based on user and assistant's message.

        :Parameters:
        user_message: Message from the user
        assistant_message: Output from the language model

        :Returns:
        Multi turn prompt for Llama-3-Instruct based on user's message and language model's output.
        """
        
        chat_history = self._generate_chat_history(user_message=user_message,
                                                   assistant_message=assistant_message)

        return self._prompt_template.safe_substitute({'system_message': self._system_message,
                                                      'chat_history': chat_history})

    def _generate_chat_history(self, user_message: str, assistant_message: str | None) -> str:
        """
        Method to generate chat history based on user and assistant's message.

        :Parameters:
        user_message: Message from the user
        assistant_message: Message generated by the language model

        :Returns:
        String representation of the chat history
        """

        if assistant_message is None:
            self._chat_history.append(f'{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>')
        else:
            self._chat_history.extend([f'{assistant_message}<|eot_id|>',
                                       '<|start_header_id|>user<|end_header_id|>'
                                       '',
                                       f'{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'])

        
        return '\n'.join(self._chat_history)
    