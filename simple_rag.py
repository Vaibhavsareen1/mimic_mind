import os
import torch
import gradio as gr
from typing import List, Tuple
from llmlib.chunkers import TextChunker
from llmlib.retrievers import Retriever
from llmlib.ingestors import PDFTextIngestor
from llmlib.text_readers import PDFTextReader
from llmlib.generators import LlamaCPPGenerator
from llmlib.prompt_templates import BasePromptTemplate
from llmlib.vectorestore_components import CustomChromaDB
from custom_utils.ui_configs import _DEFAULT_GRADIO_THEME
from custom_utils.prompt_handlers import STPromptTemplateHandler
from custom_utils.vectorstore_handlers import VectorstoreInfoHandler
from custom_utils.model_handlers import GenModelInfoHandler, RetModelInfoHandler



with gr.Blocks(theme=_DEFAULT_GRADIO_THEME, title='Simple R.A.G Application') as demo:
    """
    Gradio app to demonstrate how large language models can be used for Question Answering task on information the
    language model has not been trained on. In this simple 'Retrieval-Augmented-Generation' demo the answers will
    not be stored. The user has the option to copy the answers provided by the language model. 
    -------------------------------------------------------------------------------------------------------------------
    Constraints: 
    The RAG app can only handle PDF files.

    Future Developments:
    Allow the app to work with files of with extensions .txt, .csv, and .xlsx.
    Allow the user to adjust number of layer of LLM that are to be pushed onto the GPU.
    -------------------------------------------------------------------------------------------------------------------
    The application is split among 3 tabs namely 'Vector Store Setup', 'Current Session Setup' and 'Chatbot'
    1. 'Vector Store Setup':
    This tab allows the user to add information about a new collection or delete an existing collection within the 
    provided vector store. For a new collection to be added the user needs to provide the name of the collection,
    the retriever model that will be associated with the collection and the chunk size which will be the size of all
    of the documents that will be stored in the collection.

    Constraints:    
    1.1 Two collections with the same name can not exist.
    1.2 The retrival model and chunk size provided during the intialization of the collection will be used for any
        future ingestion or retrieval process which are associated with the selected collection.

    2. Current Session Setup:
    This tab allows the user to select the collection they want to work with, add new documents to the selected
    collection, choose their prefered language model to perform the 'generator' process, the temperature the user
    wants the language model to work with in order to induce creativity into the provided answers and the documents the
    user wants to ingest.

    The user will be able to see the following (The purpose is to show what the user is working with):
    - The retriever model's name associated with the collection.
    - The chunk size of documents stored in the collection.
    - Names of all documents ingested in the collection.
    - Length of context window of the large language model selected by the user.

    Constraints:
    2.1 Duplicate document will not be ingested.

    3. Chatbot:
    This tab allows the user ask questions and obtain answers from the LLM based on the external knowledge ingested
    in the current session's collection.
    """

    # Variables to be used through out the gradio app
    # File paths 
    _SOURCE_DIR = os.getcwd()
    _GENERATOR_CONFIG_FILE_PATH = os.path.join(_SOURCE_DIR, 'model_configs', 'generator_config.json')
    _RETRIEVER_CONFIG_FILE_PATH = os.path.join(_SOURCE_DIR, 'model_configs', 'retriever_config.json')
    _VECTORSTORE_CONFIG_FILE_PATH = os.path.join(_SOURCE_DIR, 'vectorstore_configs', 'vectorstore_config.json')
    _DOC_STORE = os.path.join(_SOURCE_DIR, 'document_storage')
    _VECTORSTORE_PATH = os.path.join(_SOURCE_DIR, 'vectorstore')

    # Create the document storage and vectorstore directory if they dont already exist
    if not os.path.isdir(_DOC_STORE):
        os.mkdir(_DOC_STORE)
    if not os.path.isdir(_VECTORSTORE_PATH):
        os.mkdir(_VECTORSTORE_PATH)

    # Handlers
    generator_handler = GenModelInfoHandler(model_config_path=_GENERATOR_CONFIG_FILE_PATH)
    retriever_handler = RetModelInfoHandler(model_config_path=_RETRIEVER_CONFIG_FILE_PATH)
    vectorstore_handler = VectorstoreInfoHandler(vectorstore_config_path=_VECTORSTORE_CONFIG_FILE_PATH)
    prompt_handler = STPromptTemplateHandler()

    # Dynamic variables that will used through out the gradio app
    collection_names = vectorstore_handler.get_collection_list()
    vectorstore_handler.switch_collection(collection_name=collection_names[0])
    ingested_documents = vectorstore_handler.get_documents()
    retriever: Retriever | None = None
    generator: LlamaCPPGenerator | None = None
    prompt_template: BasePromptTemplate | None = None

    # Static variables that are to be instantiated only once
    retriever_names: List[str] = retriever_handler.get_model_list()
    generator_names: List[str] = generator_handler.get_model_list()
    retriever_name = retriever_handler.get_model_name()
    retriever_max_tokens = retriever_handler.get_max_tokens()
    _DEFAULT_SYSTEM_MESSAGE = ' '.join(['You are a responsible AI who only provides answers based on the content provided to you.',
                                        'You and do not add any additional information from your end.'])

    with gr.Tab(label='Vector Store Setup') as vectorstore_config_tab:
        # Add phase where gradio components for adding collection name, choice of retriever model, chunk size
        # and overlap size is configured and saved
        with gr.Column() as config_addition_column:
            # Gradio Row component handling addition of a collection's name
            with gr.Row() as collection_name_row:
                collection_name_text = gr.Textbox(value=' '.join(['Please add a new collection name here.',
                                                                  'The name you provide should not already exist.']),
                                                  interactive=False,
                                                  show_label=False,
                                                  scale=2)
                
                collection_name_comp = gr.Textbox(interactive=True,
                                                  show_label=False,
                                                  scale=1)
            
            # Gradio Row component handling selection of preferred retriever model
            with gr.Row() as retriever_selection_row:
                retriever_name_text = gr.Textbox(label=None,
                                                 value='Select your choice of retriver.',
                                                 interactive=False,
                                                 show_label=False,
                                                 scale=2)
                
                retriever_name_comp = gr.Dropdown(value=retriever_name,
                                                  choices=retriever_names,
                                                  interactive=True,
                                                  multiselect=False,
                                                  show_label=False,
                                                  scale=1)

            # Gradio Row component handling chunk size selection    
            with gr.Row() as chunk_size_row:
                chunk_size_text = gr.Textbox(value=' '.join(['The chunk size will always be between 1 and maximum',
                                                             'context window of the chosen retriever.',
                                                             ]),
                                             interactive=False,
                                             show_label=False,
                                             scale=2)
                
                chunk_size_comp = gr.Slider(minimum=1,
                                            maximum=retriever_max_tokens,
                                            value=retriever_max_tokens // 2,
                                            interactive=True,
                                            show_label=False,
                                            scale=1)
            
            # Gradio Row component handling overlap size used during ingestion phase
            with gr.Row() as chunk_size_row:
                overlap_size_text = gr.Textbox(value=' '.join(['The overlap size will always be between 1 and maximum',
                                                               'context window of the chosen retriever.']),
                                             interactive=False,
                                             show_label=False,
                                             scale=2)
                
                overlap_size_comp = gr.Slider(minimum=1,
                                              maximum=retriever_max_tokens,
                                              value=retriever_max_tokens // 2,
                                              interactive=True,
                                              show_label=False,
                                              scale=1)

            # Gradio Button component which once triggered will perform the following steps
            # 1. Add the collection and its information into the database
            # 2. Update the choices available in the 'Current Session Setup' tab
            # 3. Clears the text display of the `collection_name_comp`
            collection_add_button = gr.Button(value='Add Collection')

        # Deletion phase where the used gets to delete an existing collection
        with gr.Column() as config_deletion_tab:
            # Gradio Row component which handles which collection needs to be deleted.
            with gr.Row() as deletion_row:
                collection_deletion_text = gr.Textbox(value='Choose the collection you want to delete',
                                                      interactive=False,
                                                      show_label=False,
                                                      scale=2)
                
                collection_deletion_comp = gr.Dropdown(value=collection_names[0],
                                                       choices=collection_names,
                                                       interactive=True,
                                                       multiselect=False,
                                                       show_label=False,
                                                       scale=1)
        
            # Gradio Button component which once triggered will perform the following steps
            # 1. Delete the collection and its information into the database
            # 2. Update the choices available in the 'Current Session Setup' tab
            deletion_button = gr.Button(value='Delete Collection')
    
    with gr.Tab(label='Current Session Setup') as current_session_config_tab:
        with gr.Row() as collection_selection_row:
            collection_selected_text = gr.Textbox(interactive=False,
                                                  show_label=False,
                                                  value=' '.join(['Choose the collection where documents are to be ingested and used for RAG Session.']),
                                                  lines=3,
                                                  scale=2)

            collection_selected_comp = gr.Dropdown(value=collection_names[0],
                                                   choices=collection_names,
                                                   interactive=True,
                                                   label='Select Collection',
                                                   scale=1)

            collection_documents_display = gr.Dropdown(label='Ingested Documents',
                                                       value=ingested_documents[0],
                                                       choices=ingested_documents,
                                                       scale=2)

            collection_retriever_display = gr.Textbox(label='Retriever',
                                                      value=vectorstore_handler.get_retriever_model(),
                                                      interactive=False,
                                                      scale=1)
            
            collection_chunk_size_display = gr.Textbox(label='Chunk Size',
                                                       value=vectorstore_handler.get_chunk_size(),
                                                       interactive=False,
                                                       scale=1)
            
            collection_overlap_size_display = gr.Textbox(label='Overlap Size',
                                                         value=vectorstore_handler.get_overlap_size(),
                                                         interactive=False,
                                                         scale=1)
    
        with gr.Row() as document_ingestion_row:
            with gr.Column() as ingestion_info_column:
                document_ingestion_text = gr.Textbox(interactive=False,
                                                     show_label=False,
                                                     value=' '.join(['Select documents you want to ingest.',
                                                                     'Duplicate files will not be ingested.']))
                single_text_comp = gr.Radio(label='Store content as a single string.',
                                            choices=['Yes', 'No'],
                                            value='Yes')
                document_ingestion_button = gr.Button(value='Ingest Documents')
            
            document_ingestion_comp = gr.File(label='Select documents',
                                              interactive=True,
                                              file_count='multiple')
            
        with gr.Row() as generator_selection_row:
            generator_selection_text = gr.Textbox(interactive=False,
                                                  show_label=False,
                                                  value='Choose the Large Language Model (LLM) you want to work with.')

            generator_selection_comp = gr.Dropdown(value=generator_names[0],
                                                   choices=generator_names,
                                                   interactive=True,
                                                   show_label=False)

        with gr.Row() as system_message_row:
            system_message_text = gr.Textbox(interactive=False,
                                             show_label=False,
                                             value='Set your own system message to be used with your language model.')
    
            system_message_comp = gr.Textbox(interactive=True,
                                             show_label=False,
                                             value=_DEFAULT_SYSTEM_MESSAGE,
                                             lines=3,
                                             )

        with gr.Row() as gpu_setting_row:
            gpu_setting_text = gr.Textbox(interactive=False,
                                          show_label=False,
                                          value='Select the number of GPU layers you want to offload onto your GPU.')
            
            gpu_setting_comp = gr.Slider(interactive=True,
                                         minimum=0,
                                         maximum=100,
                                         step=1,
                                         value=12,
                                         show_label=False)

        with gr.Row() as temperature_setting_row:
            temperature_setting_text = gr.Textbox(interactive=False,
                                                  show_label=False,
                                                  value=' '.join(['Select the temperature to induce creativity.',
                                                                  'Larger the value more creative the model']))

            temperature_setting_comp = gr.Slider(interactive=True,
                                                 minimum=0,
                                                 maximum=1,
                                                 step=0.01,
                                                 show_label=False)

        with gr.Row() as document_retrival_row:
            document_retrival_text = gr.Textbox(interactive=False,
                                                show_label=False,
                                                value='Select the number of documents to be retrieved.')
            
            document_retrival_comp = gr.Slider(interactive=True,
                                               minimum=1,
                                               maximum=100,
                                               step=1,
                                               show_label=False)
    
        start_session_button = gr.Button(value='Start Session')

    with gr.Tab(label='Chatbot') as qa_interface_tab:
        
        chat_history = gr.Chatbot(show_copy_button=True,
                                  render_markdown=True,
                                  show_label=False)

        with gr.Row() as user_input_row:
            user_question = gr.Textbox(interactive=True,
                                       show_label=False,
                                       scale=3)

            send_button = gr.Button(scale=1,
                                    value='Send')

    # Functions that will get triggered once a gradio component gets updated
    @collection_add_button.click(inputs=[collection_name_comp,
                                         retriever_name_comp,
                                         chunk_size_comp,
                                         overlap_size_comp],
                                 outputs=[collection_name_comp,
                                          collection_deletion_comp,
                                          collection_selected_comp,
                                          collection_documents_display,
                                          collection_retriever_display,
                                          collection_chunk_size_display,
                                          collection_overlap_size_display])
    def add_collection(collection_name: str,
                       retriever_name: str,
                       chunk_size: int,
                       overlap_size: int) -> Tuple[str, List[str], List[str], List[str], str, int, int]:
        """
        Function which gets triggered when the 'Add Collection' gradio button gets triggered. This 
        function updates the database with a collection with its related information.

        :Parameters:
        collection_name: Name of the new collection
        retriever_name: Name of the retriever model used during the ingestion and retrieval stage of RAG
        chunk_size: Chunk size of the documents that will stored in the vectordatabase
        overlap_size: Size of overlapping between two chunks to retain context of the original document.

        :Returns:
        A tuple of updated gradio components that reflect the changes made
        """
        # Call global variables into local scope
        global collection_names, ingested_documents, vectorstore_handler

        # Update the vectorstore database if the collection name is unique
        if collection_name not in collection_names:
            # Update the vectorstore database
            vectorstore_handler.add_collection(collection_name=collection_name,
                                               retriever_model_name=retriever_name,
                                               chunk_size=chunk_size,
                                               overlap_size=overlap_size)

            # Update global variables
            collection_names = vectorstore_handler.get_collection_list()
            vectorstore_handler.switch_collection(collection_name=collection_names[0])
            ingested_documents = vectorstore_handler.get_documents()
            retriever_name = vectorstore_handler.get_retriever_model()
            chunk_size = vectorstore_handler.get_chunk_size()
            overlap_size = vectorstore_handler.get_overlap_size()
        else:
            gr.Info(f'Please provide a unique collection name. {collection_name} already exists in the database.')

        # Return the updated gradio components
        return (gr.Textbox(value=None,
                           interactive=True,
                           show_label=False,
                           scale=1),
                gr.Dropdown(value=collection_names[0],
                            choices=collection_names,
                            interactive=True,
                            multiselect=False,
                            show_label=False,
                            scale=1),
                gr.Dropdown(value=collection_names[0],
                            choices=collection_names,
                            interactive=True,
                            label='Select Collection',
                            scale=1),
                gr.Dropdown(label='Ingested Documents',
                            value=ingested_documents[0],
                            choices=ingested_documents,
                            scale=2),
                retriever_name,
                chunk_size,
                overlap_size)

    @deletion_button.click(inputs=[collection_deletion_comp],
                           outputs=[collection_deletion_comp,
                                    collection_selected_comp,
                                    collection_documents_display,
                                    collection_retriever_display,
                                    collection_chunk_size_display,
                                    collection_overlap_size_display])
    def delete_collection(collection_name: str) -> Tuple[List[str], List[str], List[str], str, int, int]:
        """
        Function which gets triggered when the 'Delete Collection' gradio button gets triggered. This 
        function updates the database by deleting the selected collection and its related information

        :Parameters:
        collection_name: Name of the collection to be deleted

        :Returns:
        A tuple containing updated gradio components for collection name and its corresponding information   
        """

        # Call global variables into local scope
        global collection_names, ingested_documents, vectorstore_handler, retriever_handler

        # Delete the collection and update the dropdowns if the collection exists
        if collection_name == 'No_Collection':
            gr.Info(message='Please delete a collection that exists')
        else:
            # switch the retriever model to the retriever model related to the collection
            retriever_name = vectorstore_handler.get_retriever_model()
            retriever_handler.switch_model(model_name=retriever_name)
            # Get retriever model path and tokenizer path
            model_path = retriever_handler.get_model_path()
            tokenizer_path = retriever_handler.get_tokenizer_path()
            # Load a chromadb to delete the collection
            chroma_database = CustomChromaDB(model_path=model_path,
                                             tokenizer_path=tokenizer_path,
                                             db_path=_VECTORSTORE_PATH,
                                             collection_name=collection_name)
            chroma_database.delete_collection(collection_name=collection_name)
            # Update the vectorstore_handler by deleting the collection and its information from it
            vectorstore_handler.delete_collection(collection_name=collection_name)
    
        collection_names = vectorstore_handler.get_collection_list()
        ingested_documents = vectorstore_handler.get_documents()
        retriever_name = vectorstore_handler.get_retriever_model()
        chunk_size = vectorstore_handler.get_chunk_size()
        overlap_size = vectorstore_handler.get_overlap_size()

        # Return updated gradio components
        return (gr.Dropdown(value=collection_names[0],
                            choices=collection_names,
                            interactive=True,
                            multiselect=False,
                            show_label=False,
                            scale=1),
                gr.Dropdown(value=collection_names[0],
                            choices=collection_names,
                            interactive=True,
                            label='Select Collection',
                            scale=1),
                gr.Dropdown(value=ingested_documents[0],
                            choices=ingested_documents,
                            interactive=True,
                            label='Ingested Documents',
                            scale=2),
                retriever_name,
                chunk_size,
                overlap_size
                )

    @retriever_name_comp.change(inputs=[retriever_name_comp],
                                outputs=[chunk_size_comp, overlap_size_comp])
    def update_chunk_overlap_size(retriever_name: str) -> Tuple[int, int]:
        """
        Function that gets triggered when the gradio component that allows user to choose which retriever is required
        for the collection gets triggered updating the chunk size and overlap size gradio components .

        :Parameters:
        retriever_name: User's preferred retriever

        :Returns:
        Tuple containing updated gradio component for chunk and overlap size
        """
        
        retriever_handler.switch_model(model_name=retriever_name)

        max_tokens = retriever_handler.get_max_tokens()

        return (gr.Slider(minimum=1,
                          maximum=max_tokens,
                          value=max_tokens // 2,
                          interactive=True,
                          show_label=False,
                          scale=1),
                gr.Slider(minimum=1,
                          maximum=max_tokens,
                          value=max_tokens // 2,
                          interactive=True,
                          show_label=False,
                          scale=1))

    # Functions that will update gradio components
    @collection_selected_comp.change(inputs=[collection_selected_comp],
                                     outputs=[collection_documents_display,
                                              collection_retriever_display,
                                              collection_chunk_size_display,
                                              collection_overlap_size_display])
    def change_collection(collection_name: str) -> Tuple[List[str], str, int, int]: 
        """
        Method to update gradio components that display the current collection and information related to it for the
        user to understand which retriever is going to be used, what the document's chunk size and overlap size would
        be. It will also update which all documents have been ingested

        :Parameters:
        collection_name: Name of the collection for which information about its retriever, chunk size and overlap size
                         will be displayed

        :Returns:
        Tuple containing updated gradio components for ingested documents, retriever name, chunk size and overlap size
        """

        # Call in variables from global scope to local scope to be updated
        global ingested_documents, vectorstore_handler

        # Update global variables
        vectorstore_handler.switch_collection(collection_name=collection_name)
        ingested_documents = vectorstore_handler.get_documents()

        # Get retriever name, chunk size and overlap size associated with the collection name
        retriever_name = vectorstore_handler.get_retriever_model()
        chunk_size = vectorstore_handler.get_chunk_size()
        overlap_size = vectorstore_handler.get_overlap_size()

                
        return (gr.Dropdown(value=ingested_documents[0],
                            choices=ingested_documents,
                            interactive=True,
                            label='Ingested Documents',
                            scale=2),
                retriever_name,
                chunk_size,
                overlap_size)

    @document_ingestion_button.click(inputs=[document_ingestion_comp, single_text_comp],
                                     outputs=[collection_documents_display, document_ingestion_comp])
    def ingest_documents(document_list: List[str], single_text: str) -> Tuple[List[str], None]:
        """
        Method to implement ingestion process in RAG methodology

        :Parameters:
        document_list: List of documents to be ingested
        single_text: Store content of the document as a single string
        
        :Returns:
        Tuple containing updated gradio components for document display and document ingestion component
        """

        # Call in global variables into local scope
        global retriever_handler, vectorstore_handler, _DOC_STORE, _VECTORSTORE_PATH

        retriever_model_name = vectorstore_handler.get_retriever_model()
        ingested_documents = vectorstore_handler.get_documents()
        retriever_handler.switch_model(model_name=retriever_model_name)

        retriever_path = retriever_handler.get_model_path()
        tokenizer_path = retriever_handler.get_tokenizer_path()
        collection_name = vectorstore_handler.get_collection_name()
        chunk_size = vectorstore_handler.get_chunk_size()
        overlap_size = vectorstore_handler.get_overlap_size()

        single_text_bool = True if single_text == 'Yes' else False
        # Instantiate a document ingestor and chunker
        chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)
        ingestor = PDFTextIngestor(model_path=retriever_path,
                                   tokenizer_path=tokenizer_path,
                                   db_path=_VECTORSTORE_PATH, 
                                   collection_name=collection_name,
                                   chunker=chunker,
                                   text_reader=PDFTextReader(single_text=single_text_bool))

        # Raise an info banner if no document is being ingested
        if (document_list is None) or (len(document_list) < 1):
            gr.Info('No document was selected. Please select a document for ingestion')
        else:
            # The files loaded in the gradio file component are stored in temp gradio folder
            # Files from the temp folder (old_document_list) will move to a central folder (new_document_list)
            old_document_list = document_list
            new_document_list = list()
            # Loop through old files
            for old_document_name in old_document_list:
                base_name = os.path.basename(old_document_name)
                new_document_name = os.path.join(_DOC_STORE, base_name)
                # Move files from old location to the new location and add them to the `new_document_list`
                os.replace(old_document_name, new_document_name)
                new_document_list.append(new_document_name)

            # Load the new documents into the collection
            ingested_documents = vectorstore_handler.get_documents()
            # Loop through the new files
            for document in new_document_list:
                # Save the base name of the document in the vectorstore information file
                document_base = os.path.basename(document)
                if document_base in ingested_documents:
                    gr.Info(f'The document {os.path.basename(document_base)} already exists in the collection')
                else:
                    # Add document in the document list and into the vectostore against the collection name
                    vectorstore_handler.add_document(document_name=document_base)
                    ingested_documents = vectorstore_handler.get_documents()
                    ingestor.ingest_document(document)
            gr.Info('All new documents have been ingested')

        return (gr.Dropdown(value=ingested_documents[0],
                            choices=ingested_documents,
                            interactive=True,
                            label='Ingested Documents',
                            scale=2),
                gr.File(value=None,
                        label='Select documents',
                        interactive=True,
                        file_count='multiple')
                )
 
    @start_session_button.click(inputs=[collection_selected_comp,
                                        generator_selection_comp,
                                        system_message_comp,
                                        gpu_setting_comp,
                                        temperature_setting_comp,
                                        document_retrival_comp],
                                outputs=[chat_history,
                                         user_question])
    def start_session(collection_name: str,
                      generator_name: str,
                      system_message: str,
                      gpu_layers: int,
                      temperature: float,
                      k_documents: int) -> Tuple[List[Tuple[str, str]], str]:
        """
        Function to load the components required to run the R.A.G framework. This function will update
        global variables that take care of vectorstore, retriever model, generator model and the prompt 
        template to be used during the chatting phase.

        This function also refreshes components in the 'Chatbot' tab 

        :Parameters:
        collection_name: Name of the collection from which documents will be retrieved
        generator_name: Name of the generator to be used for the 'generation process'
        system_message: System message to be set to be used in the prompt template
        gpu_setting_comp: Number of layers to offload onto the GPU
        temperature: Temperature value to improve the creativity
        k_documents: Number of documents that are to be retrieved during the retrieval process
        
        :Returns:
        None
        """

        # Call in variables from global to local scope
        global retriever_handler, generator_handler, vectorstore_handler, prompt_handler, _VECTORSTORE_PATH
        global retriever, generator, prompt_template

        # Switch to the current collection_name
        vectorstore_handler.switch_collection(collection_name=collection_name)
        # Switch to the retriever connected to the current collection
        retriever_name = vectorstore_handler.get_retriever_model()
        retriever_handler.switch_model(model_name=retriever_name)
        retriever_model_path = retriever_handler.get_model_path()
        retriever_tokenizer_path = retriever_handler.get_tokenizer_path()

        # Switch to the user's requested generator
        generator_handler.switch_model(model_name=generator_name)
        generator_model_path = generator_handler.get_model_path()
        generator_max_tokens = generator_handler.get_context_length()

        # Set the system message to None if the system message provided by the user is empty
        system_message = system_message if ((isinstance(system_message, str)) and (len(system_message) > 0)) else None
    
        # Set the new retriever and generator
        retriever = Retriever(model_path=retriever_model_path,
                              tokenizer_path=retriever_tokenizer_path,
                              db_path=_VECTORSTORE_PATH,
                              collection_name=collection_name,
                              top_k=k_documents)

        generator = LlamaCPPGenerator(model_path=generator_model_path,
                                      max_tokens=generator_max_tokens,
                                      temperature=temperature,
                                      n_gpu_layers=gpu_layers if torch.cuda.is_available() else 0,
                                      context_size=generator_max_tokens)
        # Set the new prompt template based on the model being used
        prompt_template = prompt_handler.get_prompt_template(model_name=generator_name,
                                                             system_message=system_message)

        # Display information about the session being started.
        gr.Info("Session has started.")

        return ([(None, None)], None)

    @send_button.click(inputs=[chat_history,
                               user_question],
                       outputs=[chat_history,
                                user_question])
    def generate(chat_history: List[Tuple[str, str]],
                 user_input: str) -> Tuple[List[Tuple[str, str]], str]:
        """
        Function to generate outputs based on user's input

        :Parameters:
        chat_history: History of the conversation between language model and the user.
        user_input: Question asked by the user

        :Returns:
        Tuple containing updated chat history and blank user input gradio component
        """

        # Pull variables from global variables into local context
        global retriever, generator, prompt_template

        # Retrieve documents based on user's input
        retrieved_text: List[str] = retriever.query(query=user_input)

        # Loop through the list of retrieved documents and generate a response
        FIRST_TEXT = True
        COUNTER = 1
        for text in retrieved_text:
            user_message = '\n'.join([user_input, '', 'Content:', text])
            prompt = prompt_template.generate_prompt(user_message=user_message)

            generated_text = generator.generate(prompt=prompt)

            if FIRST_TEXT:
                chat_history.append((user_input, f'Response based on Retrieved Document {COUNTER}:\n {generated_text}'))
                FIRST_TEXT = False
            else:
                chat_history.append((None, f'Response based on Retrieved Document {COUNTER}:\n {generated_text}'))
        
            COUNTER += 1
        return (chat_history,
                None)

if __name__ == '__main__':
    demo.launch()