import os 
import torch
import gradio as gr
from custom_utils.ui_configs import _DEFAULT_GRADIO_THEME
from custom_utils.model_handlers import GenModelInfoHandler
from custom_utils.prompt_handlers import STPromptTemplateHandler
from typing import List
from llmlib.text_readers import PDFTextReader
from llmlib.summarizers import IncrementalUpdater, HierarchicalSummarizer
from llmlib.generators import LlamaCPPGenerator, OpenAISingleTurnGenerator



with gr.Blocks(theme=_DEFAULT_GRADIO_THEME, title='Summarization Application') as demo:
    """
    Gradio app to demonstrate how large language models can be used for summarization task on documents that are longer
    than average context length of large language models. In this simple 'Summarizatoin' demonstration the summaries
    will not be stored. The user has the option to copy the summaries provided by the language model. Summary length 
    will be constrained between ~100 tokens and 3/8th of the context length.

    Available summarization techniques -
    1. Hierarchical Summarization
    2. Incremental Updating

    Summarization techniques have been taken from the papers -
    1. BooookScore: A systematic exploration of book-length summarization in the era of LLMs
    -------------------------------------------------------------------------------------------------------------------
    Constraints: 
    The Summarization application can only handle PDF files.
    The content of the document will not be clubbed into a single text for summarization

    Future Developments:
    Allow the app to work with files of with extensions .txt, .csv, .xlsx and other document formats.
    -------------------------------------------------------------------------------------------------------------------
    The application is a single tab application. Within the same tab the user has the ability to -
    Inputs
    1. Add documents to be summarized
    2. Choose the Large Language Model which will perform summarization task.
    3. Choose the length of the summary.
    4. Choose the type of summarization technique.
    5. Choose the temperature to induce creativity in summarization process.
    6. Choose what type of document is being summarized.

    Outputs
    1. Get document summaries
    """

    # Global variables
    # File paths 
    _SOURCE_DIR = os.getcwd()
    _GENERATOR_CONFIG_FILE_PATH = os.path.join(_SOURCE_DIR, 'model_configs', 'generator_config.json')

    # Handlers
    generator_handler = GenModelInfoHandler(model_config_path=_GENERATOR_CONFIG_FILE_PATH)
    prompt_handler = STPromptTemplateHandler()

    # Static variables that are to be instantiated only once
    generator_names: List[str] = generator_handler.get_model_list()
    summarization_methodologies = ['Hierarchical Summarization', 'Incremental Updating']
    document_types = ['Book', 'Research Paper']

    # Dynamic variables
    generator_api_key: str = generator_handler.get_api_key()
    generator_max_tokens = generator_handler.get_context_length()
    generator_model_path = generator_handler.get_model_path()
    generator_model_type = generator_handler.get_model_type()

    _DEFAULT_SYSTEM_MESSAGE = ' '.join(['You are a Summarization Specialist.',
                                        'Your role is to extract key information and generate concise, coherent summaries from various content types.',
                                        'As a summarization specialist you make sure you are summarizing content according to word limit and at the same time you are not adding any additional information from you end.'])

    with gr.Tab(label='Summarizer') as summarizer_tab:
        """
        This summarizer_tab contains all the input and output components that the user needs to work with in order to
        perform summarization task on a document.
        """

        with gr.Row() as model_information_row:
            """
            Row component containing model and its related information components that the user needs to select before
            beginning the summarization process. This row component is responsible for -
            1. Large Language Model (LLM) name selection.
            2. Temperature Selection
            3. Number of model layers to offload to the GPU
            """

            model_name_component = gr.Dropdown(choices=generator_names,
                                               interactive=True,
                                               label='Large Language Model',
                                               type='value',
                                               value=generator_names[0])
            
            temperature_component = gr.Slider(interactive=True,
                                              minimum=0,
                                              maximum=1,
                                              label='Temperature',
                                              step=0.01,
                                              value=0)

            gpu_layer_component = gr.Slider(interactive=True,
                                            minimum=0,
                                            maximum=100,
                                            label='Layers to offload',
                                            step=1,
                                            value=12)
        
        with gr.Row() as summarization_info_row:
            """
            Row component containing summarization process and its related information components that the user needs
            to select before beginning the summarization process. This row component is responsible for -
            1. Summarization Methology
            2. Type of Document
            3. Required summary length
            4. Component to load documents
            """

            with gr.Column() as info_column:
                methodology_component = gr.Dropdown(choices=summarization_methodologies,
                                                    interactive=True,
                                                    label='Methology',
                                                    multiselect=False,
                                                    type='value',
                                                    value=summarization_methodologies[0])
                
                document_type_component = gr.Dropdown(choices=document_types,
                                                    interactive=True,
                                                    label='Document Type',
                                                    multiselect=False,
                                                    type='value',
                                                    value=document_types[0])
    
                summary_length_component = gr.Slider(minimum=0,
                                                     maximum=int((generator_max_tokens * 3) / 8),
                                                     label='Summary Length',
                                                     value=int((100 + ((generator_max_tokens * 3) / 8)) // 2),
                                                     interactive=True,
                                                     step=50)

                document_ingestion_component = gr.File(label='Load documents',
                                                       interactive=True,
                                                       file_count='multiple',
                                                       file_types=['.pdf'])
    
            # Gradio text box which will display the summaries
            summary_box_component = gr.Textbox(label='Summary Display',
                                               interactive=False,
                                               show_copy_button=True,
                                               scale=5)

        # Gradio button which will start the process of summarization
        summarization_button = gr.Button(interactive=True,
                                         value='Summarize')

    @summarization_button.click(inputs=[document_ingestion_component, document_type_component,
                                        model_name_component, methodology_component,
                                        gpu_layer_component, summary_length_component,
                                        temperature_component],
                                outputs=[summary_box_component])
    def summarize(document_list: List[str] | None,
                  document_type: str,
                  generator_name: str,
                  methodology: str,
                  offloading_layers: int,
                  summary_length: int,
                  temperature: float
                  ) -> str:
        """
        Function to summarize a document based on the method chosen by the user

        :Parameters:
        document_list: A list containing documents to be summarized.
        document_type: Document type to summarize.
        generator_name: Name of the model to use for the summarization process.
        methodology: Name of the summarization technique to use for summarization.
        offloading_layers: Number of model layers to offload to the GPU incase an open-source model is 
                           being used.
        summary_length: Length of the final summary
        temperature: Temperature value to induce creativity in the summarization process.

        :Returns:
        A single string containing summaries for all of the documents. 
        """

        # Call in global variables
        global generator_handler, prompt_handler, _DEFAULT_SYSTEM_MESSAGE

        # If no document has been provided raise a warning that no documents were provided
        if document_list is None:
            gr.Info('Please provide a document to be summarized.')

            return None

        # Load text readers
        reader = PDFTextReader(single_text=False)

        # Load generators and their corresponding prompt template based on the model type.
        generator_handler.switch_model(model_name=generator_name)
        generator_api_key = generator_handler.get_api_key()
        generator_max_tokens = generator_handler.get_context_length()
        generator_model_path = generator_handler.get_model_path()
        generator_model_type = generator_handler.get_model_type()

        if generator_model_type == 'Open-Source':
            generator = LlamaCPPGenerator(model_path=generator_model_path,
                                          context_size=generator_max_tokens,
                                          max_tokens=generator_max_tokens,
                                          n_gpu_layers=offloading_layers if torch.cuda.is_available() else 0,
                                          temperature=temperature)
                                          
            # Set the new prompt template based on the model being used
            prompt_template = prompt_handler.get_prompt_template(model_name=generator_name,
                                                                 system_message=_DEFAULT_SYSTEM_MESSAGE)
        else:
            generator = OpenAISingleTurnGenerator(api_key=generator_api_key,
                                                  model_name=generator_name,
                                                  number_of_responses=1,
                                                  system_message=_DEFAULT_SYSTEM_MESSAGE,
                                                  temperature=temperature)
            prompt_template = None

        # Load the summarizer based on the methodology selected and type of documents being loaded
        if methodology == 'Incremental Updating':
            summarizer = IncrementalUpdater(generator=generator,
                                            document_type=document_type,
                                            model_prompt_template=prompt_template,
                                            model_type=generator_model_type,
                                            summary_length=summary_length)
        else:
            summarizer = HierarchicalSummarizer(generator=generator,
                                                document_type=document_type,
                                                model_prompt_template=prompt_template,
                                                model_type=generator_model_type,
                                                summary_length=summary_length)

        # Loop through the list of documents and append the final summary
        summary_list : List[str] = []
        for document in document_list:
            # Get the name of the document from the document file path
            document_name = os.path.basename(document)

            # Raise a warning and do not summarize a document if it is not a PDF document
            if not document_name.lower().endswith('.pdf'):
                gr.Info(f'{document_name} is not going to be summarized as only PDF format is supported.')

                continue

            # Load the text content of the pdf from the document
            pages: List[str] = reader.convert_to_text(source_file_path=document)

            # Extract summary by using the model and passing the pages to the model
            document_summary = summarizer.summarize(document_pages=pages)
            # extend summary list with the document name and its extracted summary.
            summary_list.extend([document_name, document_summary])

        # Concatenate the summaries into a single string and return it
        final_summary_string: str = '\n--------------------\n'.join(summary_list)

        return final_summary_string

    @model_name_component.change(inputs=[model_name_component],
                                 outputs=[summary_length_component])
    def update_summary_length(model_name: str) -> int:
        """
        Function to update the summary length limits based on the model being used.

        :Parameters:
        model_name: Name of the language model

        :Returns:
        Updated summary length component
        """

        # Call in global variables
        global generator_handler

        # Switch to the latest model
        generator_handler.switch_model(model_name=model_name)

        # Get context length of the model
        generator_max_tokens = generator_handler.get_context_length()

        return gr.Slider(minimum=0,
                         maximum=int((generator_max_tokens * 3) / 8),
                         label='Summary Length',
                         value=int((100 + ((generator_max_tokens * 3) / 8)) // 2),
                         interactive=True,
                         step=50)


if __name__ == '__main__':
    demo.launch()