from string import Template
from typing import List, Literal
from llmlib.generators import BaseGenerator
from llmlib.prompt_templates import BasePromptTemplate

class BaseSummarizer(object):
    """
    Base class of summarizer that needs to be inherited by all summarizers
    """

    def __init__(self) -> None:
        """
        Method to instantiate object of :class: BaseSummarizer

        :Parameters:
        None

        :Returns:
        Instance of :class: BaseSummarizer
        """

        super().__init__()

    def summarize(self) -> None:
        """
        Method to summarize a document which every subclass of :class: BaseSummarizer
        needs to be override and implement

        :Parameters:
        None

        :Returns:
        None
        """

        pass 


class HierarchicalSummarizer(BaseSummarizer):
    """
    Class to implement 'Hierarchical Summarization' methodology to summarize text which are longer than the context
    window of a large language model. This methodology comes from the paper -
    'BooookScore: A systematic exploration of book-length summarization in the era of LLMs'

    The class currently summarizes novels and research papers only.
    """

    def __init__(self,
                 generator: BaseGenerator,
                 document_type: Literal['Book', 'Research Paper'] = 'Book',
                 model_prompt_template: BasePromptTemplate | None = None,
                 model_type: Literal['Open-Source', 'OpenAI'] = 'Open-Source',
                 summary_length: int = 2000
                 ) -> None:
        """
        Method to instantiate object of :class: HierarchicalSummarizer

        :Parameters:
        generator: Language model preferred by the user.
        document_type: Document type to be summarized.
        model_prompt_template: Prompt template used by the generator.
        model_type: Type of language model being used for summarization process.
        summary_length: Length of the final summary
        
        :Returns:
        Instance of :class: HierarchicalSummarizer
        """

        super().__init__()

        # store parameters as private attributes
        self._generator = generator
        self._model_prompt_template = model_prompt_template
        self._model_type = model_type
        self._summary_length = summary_length

        # Avaiable summarization prompts
        self._available_artifact_removal_prompts = {
            'Book': Template('\n'.join(["Below is a summary of a book:",
                                        "",
                                        "---",
                                        "",
                                        "${page_content}",
                                        "",
                                        "---",
                                        "",
                                        " ".join(["Your task is to edit the book summary by removing any phrases that indicate it was developed progressively.",
                                                  """Delete terms such as "in the ... segment," "in ... part of the story," "in the ... excerpt," "in the updated summary," and any similar phrases.""",
                                                  "The goal is to make the summary read as if it was written all at once, not in stages.",
                                                  "In addition, eliminate any elements taken from non-narrative sections like the table of contents, acknowledgments, author’s biography, author’s note, information of the author’s other works, and so on.",
                                                  "Apart from these adjustments, do not make any other changes to the summary."])])),

            'Research Paper': Template('\n'.join(["Below is a summary of a research paper:",
                                                  "",
                                                  "---",
                                                  "",
                                                  "${page_content}",
                                                  "",
                                                  "---",
                                                  "",
                                                  " ".join(["Your task is to edit the research paper summary by removing any phrases that indicate it was developed progressively.",
                                                            """Delete terms such as "in the ... section," "in ... part of the paper," "in the ... excerpt," "in the updated summary," and any similar phrases.""",
                                                            "The goal is to make the summary read as if it was written all at once, not in stages.",
                                                            "In addition, eliminate any elements taken from non-research sections like the acknowledgments, author’s biography, author's note, information about the author’s other works, and so on.",
                                                            "Apart from these adjustments, do not make any other changes to the summary."])]))}
        
        self._available_low_level_prompts = {
            'Book': Template('\n'.join(["Below is a part of a story:",
                                        "",
                                        "---",
                                        "",
                                        "${page_content}",
                                        "",
                                        "---",
                                        "",
                                        " ".join(["We are creating one comprehensive summary for the story by recursively merging summaries of its chunks.",
                                                  "Now, write a summary for the excerpt provided above, make sure to include vital information related to key events, backgrounds, settings, characters, their objectives, and motivations.",
                                                  "You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary.",
                                                  "The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc.",
                                                  "Therefore, you should organize the summary so it presents a consistent and chronological narrative.",
                                                  "Despite this recursive merging process, you need to create a summary that seems as though it is written in one go.",
                                                  "The summary must be within ${summary_length} words and could include multiple paragraphs."]),
                                        "",
                                        "Summary:"])),

            'Research Paper': Template('\n'.join(["Below is a part of a research paper:",
                                                  "",
                                                  "---",
                                                  "",
                                                  "${page_content}",
                                                  "",
                                                  "---",
                                                  "",
                                                  " ".join(["We are creating one comprehensive summary for the research paper by recursively merging summaries of its sections.",
                                                            "Now, write a summary for the excerpt provided above, ensuring to include vital information related to the study's background, objectives, methods, findings, and conclusions.",
                                                            "Briefly introduce any significant terms or concepts mentioned for the first time in the summary.",
                                                            "The research paper may feature complex methodologies, diverse data sets, and detailed analyses.",
                                                            "Therefore, you should organize the summary to present a coherent and logical narrative.",
                                                            "Despite this recursive merging process, create a summary that seems as though it is written in one go.",
                                                            "The summary must be within ${summary_length} words and could include multiple paragraphs."]),
                                                  "",
                                                  "Summary:"]))}

        self._available_intermediate_prompts = {
            'Book': Template('\n'.join(["Below is a summary of the context preceding a part of the story:",
                                        "",
                                        "---",
                                        "",
                                        "${preceeding_summary}",
                                        "",
                                        "---",
                                        "",
                                        "Below is a summary of a consecutive part of the story:",
                                        "",
                                        "---",
                                        "",
                                        "${consecutive_summary}",
                                        "",
                                        "---",
                                        "",
                                        " ".join(["We are creating one comprehensive summary for the story by recursively merging summaries of its chunks.",
                                                  "Now, merge the preceding context and the summaries into one single summary, make sure to include vital information related to key events, backgrounds, settings, characters, their objectives, and motivations.",
                                                  "You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary.",
                                                  "The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints etc.",
                                                  "Therefore, you should organize the summary so it presents a consistent and chronological narrative.",
                                                  "Despite this recursive merging process, you need to create a summary that seems as though it is written in one go.",
                                                  "The summary must be within ${summary_length} words and could include multiple paragraphs."]),
                                        "",
                                        "Summary:"])),

            'Research Paper': Template('\n'.join(["Below is a summary of the context preceding a part of a research paper:",
                                                  "",
                                                  "---",
                                                  "",
                                                  "${preceeding_summary}",
                                                  "",
                                                  "---",
                                                  "",
                                                  "Below is a summary of a consecutive part a research paper:",
                                                  "",
                                                  "---",
                                                  "",
                                                  "${consecutive_summary}",
                                                  "",
                                                  "---",
                                                  "",
                                                  " ".join(["We are creating one comprehensive summary for the research paper by recursively merging summaries of its sections.",
                                                            "Now, merge the preceding context and the summaries into one single summary, ensuring to include vital information related to the study's background, objectives, methods, findings, and conclusions.",
                                                            "Briefly introduce any significant terms or concepts mentioned for the first time in the summary.",
                                                            "The research paper may feature complex methodologies, diverse data sets, and detailed analyses.",
                                                            "Therefore, you should organize the summary to present a coherent and logical narrative.",
                                                            "Despite this recursive merging process, create a summary that seems as though it is written in one go.",
                                                            "The summary must be within ${summary_length} words and could include multiple paragraphs."]),
                                                  "",
                                                  "Summary:"]))}

        # Current prompts obtained from available prompts
        self._current_artifact_removal_prompt = self._available_artifact_removal_prompts.get(document_type)
        self._current_intermediate_prompt = self._available_intermediate_prompts.get(document_type)
        self._current_low_level_prompt = self._available_low_level_prompts.get(document_type)

        # Initialize attributes to be used during the summarization process
        self._low_level_summaries: List[str] = []
        self._final_summary: str | None = None

    def summarize(self, document_pages: List[str]) -> str:
        """
        Method to summarize a given document

        :Parameters:
        document_pages: List of document's pages in the form of text.

        :Returns:
        Final summary of the document
        """
        
        # Generate low level summaries
        self._generate_low_level_summaries(document_pages=document_pages)
        
        # Perform hierarchical updating of summaries
        self._hierarchical_updating()

        # Perform artifact removal
        self._artifact_removal()

        # Reset object to be used for summarization of a new document and return the final summary

        final_summary = self._final_summary
        self._low_level_summaries = []
        self._final_summary = None

        return final_summary

    def _generate_low_level_summaries(self, document_pages: List[str]) -> bool:
        """
        Method to implement the first step of the method in which low level summaries of content of each page is
        generated. These 'low-level' summaries are stored in the attribute '_low_level_summaries'.

        :Parameters:
        document_pages: List of document's pages in the form of text.

        :Returns:
        Boolean representing completion of the process.
        """

        for page in document_pages:
            # Generate content for the prompt
            prompt_content = self._current_low_level_prompt.safe_substitute({'page_content': page,
                                                                             'summary_length': self._summary_length})
            # Generate low-level summary for each page
            if self._model_type == 'Open-Source':
                user_message = self._model_prompt_template.generate_prompt(user_message=prompt_content)
                page_summary: str = self._generator.generate(user_message=user_message)[0]
            else:
                page_summary: str = self._generator.generate(user_message=prompt_content)[0]
            
            self._low_level_summaries.append(page_summary)

        return True

    def _hierarchical_updating(self) -> bool:
        """
        Method to implement second step of the methodology in which hierarchical updation of low level summaries get
        generated using private method '_generate_low_level_summaries'. This step includes hierarchically updating 
        summaries and merging summaries with prior context.

        :Parameters:
        None

        :Returns:
        Boolean representing completion of the hierarchical updating. 
        """

        summary_count: int = len(self._low_level_summaries)
        # Hierarchically update the low level summaries till only 1 summary is left
        # Save these summaries at _low_level_summaries itself for further use. 
        while summary_count > 1:
            temp_summaries: List[str] = []
            for index in range(0, summary_count, 2):
                if (summary_count % 2 == 1) and (index == summary_count - 1):
                    temp_summaries.append(self._low_level_summaries[index])
                else:
                    prompt_content = self._current_intermediate_prompt.safe_substitute({'preceeding_summary': self._low_level_summaries[index],
                                                                                        'consecutive_summary': self._low_level_summaries[index],
                                                                                        'summary_length': self._summary_length})
                    if self._model_type == 'Open-Source':
                        user_message = self._model_prompt_template.generate_prompt(user_message=prompt_content)
                        page_summary: str = self._generator.generate(user_message=user_message)[0]
                    else:
                        page_summary: str = self._generator.generate(user_message=prompt_content)[0]
                    
                    temp_summaries.append(page_summary)
            
            self._low_level_summaries = temp_summaries
            summary_count: int = len(self._low_level_summaries)
        
        return True
    
    def _artifact_removal(self) -> bool:
        """
        Method to implement the final step of the methodology in which the final summary is smoothened out to be 
        coherent for the user to understand.

        :Parameters:
        None

        :Returns:
        Boolean representing completion of the process 
        """

        prompt_content = self._current_artifact_removal_prompt.safe_substitute({'page_content': self._low_level_summaries[0]})

        if self._model_type == 'Open-Source':
            user_message = self._model_prompt_template.generate_prompt(user_message=prompt_content)
            page_summary: str = self._generator.generate(user_message=user_message)[0]
        else:
            page_summary: str = self._generator.generate(user_message=prompt_content)[0]

        # The final summary is the output we get from a artifact removal stage
        self._final_summary = page_summary

        return True


class IncrementalUpdater(BaseSummarizer):
    """
    Class to implement 'Incremental Updating' methodology to summarize text which are longer than the context
    window of a large language model. This methodology comes from the paper -
    'BooookScore: A systematic exploration of book-length summarization in the era of LLMs'

    The class currently summarizes novels and research papers only.
    """

    def __init__(self,
                 generator: BaseGenerator,
                 document_type: Literal['Book', 'Research Paper'] = 'Book',
                 model_prompt_template: BasePromptTemplate | None = None,
                 model_type: Literal['Open-Source', 'OpenAI'] = 'Open-Source',
                 summary_length: int = 2000) -> None:
        """
        Method to instantiate object of :class: IncrementalUpdater

        :Parameters:
        generator: Language model preferred by the user.
        document_type: Document type to be summarized.
        model_prompt_template: Prompt template used by the generator.
        model_type: Type of language model being used for summarization process.
        summary_length: Length of the final summary
        
        :Returns:
        Instance of :class: IncrementalUpdater
        """

        super().__init__()

        # store parameters as private attributes
        self._generator = generator
        self._model_prompt_template = model_prompt_template
        self._model_type = model_type
        self._summary_length = summary_length

        # Avaiable summarization prompts
        self._available_artifact_removal_prompts = {
            'Book': Template('\n'.join(["Below is a summary of a book:",
                                        "",
                                        "---",
                                        "",
                                        "${page_content}",
                                        "",
                                        "---",
                                        "",
                                        " ".join(["Your task is to edit the book summary by removing any phrases that indicate it was developed progressively.",
                                                  """Delete terms such as "in the ... segment," "in ... part of the story," "in the ... excerpt," "in the updated summary," and any similar phrases.""",
                                                  "The goal is to make the summary read as if it was written all at once, not in stages.",
                                                  "In addition, eliminate any elements taken from non-narrative sections like the table of contents, acknowledgments, author’s biography, author’s note, information of the author’s other works, and so on.",
                                                  "Apart from these adjustments, do not make any other changes to the summary."])])),

            'Research Paper': Template('\n'.join(["Below is a summary of a research paper:",
                                                  "",
                                                  "---",
                                                  "",
                                                  "${page_content}",
                                                  "",
                                                  "---",
                                                  "",
                                                  " ".join(["Your task is to edit the research paper summary by removing any phrases that indicate it was developed progressively.",
                                                            """Delete terms such as "in the ... section," "in ... part of the paper," "in the ... excerpt," "in the updated summary," and any similar phrases.""",
                                                            "The goal is to make the summary read as if it was written all at once, not in stages.",
                                                            "In addition, eliminate any elements taken from non-research sections like the acknowledgments, author’s biography, author's note, information about the author’s other works, and so on.",
                                                            "Apart from these adjustments, do not make any other changes to the summary."])]))}

        self._available_compression_prompts = {
            'Book': Template('\n'.join(['Below is a summary of part of a story:',
                                        "",
                                        "---",
                                        "",
                                        "${page_content}",
                                        "",
                                        "---",
                                        "",
                                        " ".join(["Currently, this summary contains ${current_summary_length} words.",
                                                  "Your task is to condense it to less than ${summary_length} words.",
                                                  "The condensed summary should remain clear, overarching, and fluid while being brief.",
                                                  "Whenever feasible, maintain details about key events, backgrounds, settings, characters, their objectives, and motivations - but express these elements more succinctly.",
                                                  "Make sure to provide a brief introduction to characters, places, and other major components during their first mention in the condensed summary.",
                                                  "Remove insignificant details that do not add much to the overall story line.",
                                                  "The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc.",
                                                  "Therefore, you should organize the summary so it presents a consistent and chronological narrative."]),
                                        "",
                                        "Condensed summary (to be within ${summary_length} words):"])),

            'Research Paper': Template('\n'.join(['Below is a summary of part of a research paper:',
                                                  '',
                                                  '---',
                                                  '',
                                                  '${page_content}',
                                                  '',
                                                  '---',
                                                  '',
                                                  ' '.join(['Currently, this summary contains ${current_summary_length} words.',
                                                            "Your task is to condense it to less than ${summary_length} words.",
                                                            "The condensed summary should remain clear, overarching, and fluid while being brief.",
                                                            "Whenever feasible, maintain details about key study components, such as background, objectives, methods, findings, and conclusions, but express these elements more succinctly.",
                                                            "Make sure to provide a brief introduction to significant terms or concepts during their first mention in the condensed summary.",
                                                            "Remove insignificant details that do not add much to the overall understanding of the research.",
                                                            "Organize the summary to present a coherent and logical narrative."]),
                                                  '',
                                                  'Condensed summary (to be within ${summary_length} words):']))}

        self._available_initial_prompts = {
            'Book': Template('\n'.join(['Below is the beginning part of a story:',
                                        "",
                                        "---",
                                        "",
                                        "${page_content}",
                                        "",
                                        "---",
                                        "",
                                        " ".join(["We are going over segments of a story sequentially to gradually update one comprehensive summary of the entire plot.",
                                                  "Write a summary for the excerpt provided above, make sure to include vital information related to key events, backgrounds, settings, characters, their objectives, and motivations.",
                                                  "You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary.",
                                                  "The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc.",
                                                  "Therefore, you should organize the summary so it presents a consistent and chronological narrative.",
                                                  "Despite this step-by-step process of updating the summary, you need to create a summary that seems as though it is written in one go.",
                                                  "The summary should roughly contain ${summary_length} words and could include multiple paragraphs."]),
                                        "",
                                        'Summary (${summary_length} words):'])),

            'Research Paper': Template('\n'.join(['Below is the beginning part of a research paper:',
                                                  '',
                                                  '---',
                                                  '',
                                                  '${page_content}',
                                                  '',
                                                  '---',
                                                  '',
                                                  ' '.join(['We are reviewing sections of a research paper sequentially to gradually updae one comprehensive summary of the tentire paper.',
                                                            "Write a summary for the excerpt provided above, ensuring to include key information related to the study's background, objectives, methods, findings, and conclusions.",
                                                            "Briefly introduce the research problem, hypothesis, and any significant terms or concepts mentioned for the first time.",
                                                            "The research paper may feature complex methodologies, diverse data sets, and detailed analyses.",
                                                            "Therefore, you should organize the summary to present a coherent and logical narrative.",
                                                            "Despite this step-by-step process of updating the summary, create a summary that seems as though it is written in one go.",
                                                            "The summary should roughly contain ${summary_length} words and could include multiple paragraphs."]),
                                                  '',
                                                  'Summary (${summary_length} words):']))}
        
        self._available_intermediate_prompts = {
            'Book': Template('\n'.join(["Below is a segment from a story:",
                                        "",
                                        "---",
                                        "",
                                        "${page_content}",
                                        "",
                                        "---",
                                        "",
                                        "Below is a summary of the story up until this point:",
                                        "",
                                        "---",
                                        "",
                                        "${summary_content}",
                                        "",
                                        "---",
                                        "",
                                        " ".join(["We are going over segments of a story sequentially to gradually update one comprehensive summary of the entire plot.",
                                                  "You are required to update the summary to incorporate any new vital information in the current excerpt.",
                                                  "This information may relate to key events, backgrounds, settings, characters, their objectives, and motivations.",
                                                  "You must briefly introduce characters, places, and other major elements if they are being mentioned for the first time in the summary.",
                                                  "The story may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc.",
                                                  "Therefore, you should organize the summary so it presents a consistent and chronological narrative.",
                                                  "Despite this step-by-step process of updating the summary, you need to create a summary that seems as though it is written in one go.",
                                                  "The updated summary should roughly contain ${summary_length} words and could include multiple paragraphs.",]),
                                        "",
                                        "Updated summary (${summary_length} words):"])),

            'Research Paper': Template('\n'.join(["Below is a segment from a research paper:",
                                                  "",
                                                  "---",
                                                  "",
                                                  "${page_content}",
                                                  "",
                                                  "---",
                                                  "",
                                                  "Below is a summary of the research paper up until this point:",
                                                  "",
                                                  "---",
                                                  "",
                                                  "${summary_content}",
                                                  "",
                                                  "---",
                                                  "",
                                                  " ".join(["We are reviewing sections of a research paper sequentially to gradually update one comprehensive summary of the entire paper.",
                                                            "You are required to update the summary to incorporate any new vital information in the current excerpt.",
                                                            "This information may relate to the study's background, objectives, methods, findings, and conclusions.",
                                                            "Briefly introduce any significant terms or concepts mentioned for the first time in the summary.",
                                                            "The research paper may feature complex methodologies, diverse data sets, and detailed analyses.",
                                                            "Therefore, you should organize the summary to present a coherent and logical narrative.",
                                                            "Despite this step-by-step process of updating the summary, create a summary that seems as though it is written in one go.",
                                                            "The updated summary should roughly contain ${summary_length} words and could include multiple paragraphs."]),
                                                  "",
                                                  "Updated summary (${summary_length} words):"]))}

        # Current prompts obtained from available prompts
        self._current_artifact_removal_prompt = self._available_artifact_removal_prompts.get(document_type)
        self._current_compression_prompt = self._available_compression_prompts.get(document_type)
        self._current_initial_prompt = self._available_initial_prompts.get(document_type)
        self._current_intermediate_prompt = self._available_intermediate_prompts.get(document_type)

        # Place holder for the final summary and its length
        self._final_summary:str | None = None
        self._final_summary_length: int = 0

    def summarize(self, document_pages: List[str]) -> str:
        """
        Method to summarize a given document

        :Parameters:
        document_pages: List of document's pages in the form of text

        :Returns:
        Final summary of the document
        """

        # Loop through the pages and generate summary.
        for page in document_pages:
            # If initial summary is not present then generate it
            if self._final_summary is None:
                self._generate_initial_summary(page_content=page)
            else:
                self._incremental_updating(page_content=page)
            
            # If the generated summary is greater than required length then compresss it.
            if self._final_summary_length > self._summary_length:
                self._compression()

        # Perform artifact removal step
        self._artifact_removal()
        # Extract final summary and reset the object in order for it to work with next set of documents
        final_summary = self._final_summary
        self._final_summary = None
        self._final_summary_length = 0

        return final_summary

    def _artifact_removal(self) -> bool:
        """
        Method to implement the final step of the methodology in which the final summary of the document is smoothened
        out for the summary to be more coherent for the user to understand.

        :Parameters:
        None

        :Returns:
        Boolean representing completion of the process
        """

        # Generate content for the prompt
        prompt_content = self._current_artifact_removal_prompt.safe_substitute({'page_content': self._final_summary})
        
        # Generate initial summary for the document using open or close source models
        if self._model_type == 'Open-Source':
            user_message = self._model_prompt_template.generate_prompt(user_message=prompt_content)
            self._final_summary: str = self._generator.generate(user_message=user_message)[0]
        else:
            self._final_summary: str = self._generator.generate(user_message=prompt_content)[0]
        
        self._final_summary_length = len(self._final_summary)

        return True

    def _compression(self) -> bool:
        """
        Method to implement the compression step of the methodology in which the summary is compressed to the length
        provided by the user.

        :Parameters:
        None

        :Returns:
        Boolean representing the completion of the process
        """

        # Generate content for the prompt
        prompt_content = self._current_compression_prompt.safe_substitute({'current_summary_length': self._final_summary_length,
                                                                           'page_content': self._final_summary,
                                                                           'summary_length': self._summary_length})
        
        # Generate initial summary for the document using open or close source models
        if self._model_type == 'Open-Source':
            user_message = self._model_prompt_template.generate_prompt(user_message=prompt_content)
            self._final_summary: str = self._generator.generate(user_message=user_message)[0]
        else:
            self._final_summary: str = self._generator.generate(user_message=prompt_content)[0]
        
        self._final_summary_length = len(self._final_summary)

        return True

    def _generate_initial_summary(self, page_content: str) -> bool:
        """
        Method to implement the first step of the methodology in which initial summary is generated

        :Parameters:
        page_content: Content of a page

        :Returns:
        Boolean representing completion of the process
        """

        # Generate content for the prompt
        prompt_content = self._current_initial_prompt.safe_substitute({'page_content': page_content,
                                                                       'summary_length': self._summary_length})
        
        # Generate initial summary for the document using open or close source models
        if self._model_type == 'Open-Source':
            user_message = self._model_prompt_template.generate_prompt(user_message=prompt_content)
            self._final_summary: str = self._generator.generate(user_message=user_message)[0]
        else:
            self._final_summary: str = self._generator.generate(user_message=prompt_content)[0]
        
        self._final_summary_length = len(self._final_summary)

        return True

    def _incremental_updating(self, page_content: str) -> bool:
        """
        Method to implement the second step of the methodology in which the initial summary gets updated
        incrementally and compressed if the summary extends beyond a certain length.

        :Parameters:
        page_content: Content of a page

        :Returns:
        Boolean representing completion of the process
        """

        # Generate content for the prompt
        prompt_content = self._current_intermediate_prompt.safe_substitute({'page_content': page_content,
                                                                            'summary_content': self._final_summary,
                                                                            'summary_length': self._summary_length})
        
        # Generate initial summary for the document using open or close source models
        if self._model_type == 'Open-Source':
            user_message = self._model_prompt_template.generate_prompt(user_message=prompt_content)
            self._final_summary: str = self._generator.generate(user_message=user_message)[0]
        else:
            self._final_summary: str = self._generator.generate(user_message=prompt_content)[0]
        
        self._final_summary_length = len(self._final_summary)

        return True