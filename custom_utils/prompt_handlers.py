import re
from typing import Dict, List, Tuple
from llmlib.prompt_templates import (BasePromptTemplate, Llama2PromptTemplate,
                                     Llama2InstructPromptTemplate, CodeLlama2InstructPromptTemplate,
                                     Llama3PromptTemplate, Llama3InstructSTPromptTemplate)


class STPromptTemplateHandler(object):
    """
    Class to switch among single-turn prompt templates based on model name. To handle new prompts this
    class will have to be manually updated. This class currently supports GGUF Models.
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: STPromptTemplateHandler.

        :Parameters:
        None

        :Returns:
        None
        """

        super().__init__()

    def get_prompt_template(self, model_name: str, system_message: str | None = None) -> BasePromptTemplate:
        """
        Method to get an instance of prompt template based on lanuage model's name and the system message.

        :Paramters:
        model_name: Name of the language model
        system_message: System message to be used by the prompt

        :Returns:
        Prompt template instance based on the language model's name 
        """
   
        formatted_model_name = self._generate_model_name(full_model_name=model_name)

        # Return the prompt class based on the formatted column name
        match formatted_model_name:
            case 'Meta-Llama-2':

                return Llama2PromptTemplate(system_message=system_message)

            case 'Meta-Llama-2-Instruct':

                return Llama2InstructPromptTemplate(system_message=system_message)

            case 'Meta-CodeLlama-2-Instruct':

                return CodeLlama2InstructPromptTemplate(system_message=system_message)

            case 'Meta-Llama-3':

                return Llama3PromptTemplate(system_message=system_message)

            case 'Meta-Llama-3-Instruct':

                return Llama3InstructSTPromptTemplate(system_message=system_message)

            case _DEFAULT:

                raise Exception(' '.join([f'{formatted_model_name} does not exist.',
                                          'You need to add it to the generator_config.json file and update the class']))

    def _generate_model_name(self, full_model_name: str) -> str:
        """
        Method to generate standardized model name to be used within the instance of the class
        to specify which model's prompt is to be generated.

        :Parameters:
        full_model_name: Name of the model which follows structure similar to names of the models present in
                        'generator_config.json'

        :Returns:
        Standarized model name to be used within the class instance
        """

        # Use regex to standardize model names
        model_name = re.sub(pattern='-\d{1,}B-Instruct', repl='-Instruct', string=full_model_name)
        model_name = re.sub(pattern='-\d{1,}B-', repl='-', string=full_model_name)
        model_name = re.sub(pattern='-GGUF.*', repl='', string=model_name)

        return model_name
