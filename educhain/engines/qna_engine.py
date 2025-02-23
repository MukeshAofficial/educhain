from typing import Optional, Type, Any, Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from educhain.core.config import LLMConfig
from educhain.models.qna_models import (
    MCQList, ShortAnswerQuestionList, TrueFalseQuestionList, FillInBlankQuestionList,
    SanityCheckSummary, SanityCheckResult,
    MultipleChoiceQuestion, ShortAnswerQuestion, TrueFalseQuestion, FillInBlankQuestion
)

QuestionType = Literal["Multiple Choice", "Short Answer", "True/False", "Fill in the Blank"]

class QnAEngine:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig()
        self.llm = self._initialize_llm(llm_config)

    def _initialize_llm(self, llm_config: LLMConfig):
        if llm_config.custom_model:
            return llm_config.custom_model
        else:
            return ChatOpenAI(
                model=llm_config.model_name,
                api_key=llm_config.api_key,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                base_url=llm_config.base_url,
                default_headers=llm_config.default_headers
            )

    def _get_response_model(self, question_type: QuestionType):
        models = {
            "Multiple Choice": MCQList,
            "Short Answer": ShortAnswerQuestionList,
            "True/False": TrueFalseQuestionList,
            "Fill in the Blank": FillInBlankQuestionList
        }
        return models.get(question_type, MCQList)

    def _get_single_response_model(self, question_type: QuestionType):
        models = {
            "Multiple Choice": MultipleChoiceQuestion,
            "Short Answer": ShortAnswerQuestion,
            "True/False": TrueFalseQuestion,
            "Fill in the Blank": FillInBlankQuestion
        }
        return models.get(question_type, MultipleChoiceQuestion)

    def generate_questions(
        self,
        topic: str,
        num: int = 1,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        **kwargs
    ) -> Any:
        response_model = response_model or self._get_response_model(question_type)
        parser = PydanticOutputParser(pydantic_object=response_model)
        format_instructions = parser.get_format_instructions()

        if prompt_template is None:
            prompt_template = """
            Generate {num} {question_type} question(s) based on the given topic.
            Topic: {topic}

            For each question, provide:
            1. The question
            2. The correct answer
            3. An explanation (optional)
            {additional_fields}

            Ensure the questions are clear, educational, and relevant to the topic.
            """

            additional_fields = {
                "Multiple Choice": "4. A list of options (including the correct answer)",
                "Short Answer": "4. A list of relevant keywords",
                "True/False": "4. The correct answer as a boolean (true/false)",
                "Fill in the Blank": "4. The word or phrase to be filled in the blank"
            }.get(question_type, "")

            prompt_template = prompt_template.replace("{additional_fields}", additional_fields)

        if custom_instructions:
            prompt_template += f"\n\nAdditional Instructions:\n{custom_instructions}"

        prompt_template += "\n\nThe response should be in JSON format.\n{format_instructions}"

        question_prompt = PromptTemplate(
            input_variables=["num", "topic", "question_type"],
            template=prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

        question_chain = question_prompt | self.llm
        results = question_chain.invoke({"num": num, "topic": topic, "question_type": question_type, **kwargs})
        results = results.content

        try:
            structured_output = parser.parse(results)
            return structured_output
        except Exception as e:
            print(f"Error parsing output in generate_questions: {e}")
            print("Raw output:", results)
            return response_model(questions=[])  # Return empty list on error

    def _correct_question(
        self,
        failed_result: SanityCheckResult,
        question_type: QuestionType,
        topic: str,
        llm: Optional[Any] = None
    ) -> Any:
        """
        Correct a failed question by passing it back to the LLM with failure feedback.
        """
        llm_to_use = llm if llm is not None else self.llm
        response_model = self._get_single_response_model(question_type)
        parser = PydanticOutputParser(pydantic_object=response_model)
        format_instructions = parser.get_format_instructions()

        prompt_template = """
        The following {question_type} question failed a sanity check. Please revise it based on the feedback provided.

        Original Question:
        Question: {question}
        Answer: {answer}
        {additional_fields}

        Sanity Check Feedback:
        - Answer Correctness: {answer_correctness}
        - Question Clarity: {question_clarity}
        - Distractor Plausibility: {distractor_plausibility}

        Topic: {topic}

        Revise the question to:
        1. Ensure the answer is correct
        2. Make the question clear and grammatically correct
        3. {revision_guidance}
        4. Provide an explanation if not already present

        Output the revised question in JSON format:
        {format_instructions}
        """

        additional_fields = ""
        if failed_result.options:
            additional_fields = f"Options:\n" + "\n".join(failed_result.options)
        elif failed_result.keywords:
            additional_fields = f"Keywords: {', '.join(failed_result.keywords)}"

        revision_guidance = {
            "Multiple Choice": "Ensure distractors are plausible but clearly wrong",
            "Short Answer": "Ensure keywords are relevant and comprehensive",
            "True/False": "Ensure the statement is unambiguous",
            "Fill in the Blank": "Ensure the blank fits naturally and has a unique correct answer"
        }.get(question_type, "")

        prompt = PromptTemplate(
            input_variables=["question", "answer", "additional_fields", "answer_correctness",
                            "question_clarity", "distractor_plausibility", "topic", "question_type"],
            template=prompt_template,
            partial_variables={"format_instructions": format_instructions, "revision_guidance": revision_guidance}
        )

        correction_chain = prompt | llm_to_use
        results = correction_chain.invoke({
            "question": failed_result.question,
            "answer": failed_result.answer,
            "additional_fields": additional_fields,
            "answer_correctness": failed_result.answer_correctness,
            "question_clarity": failed_result.question_clarity,
            "distractor_plausibility": failed_result.distractor_plausibility,
            "topic": topic,
            "question_type": question_type
        })
        results = results.content

        try:
            corrected_question = parser.parse(results)
            return corrected_question
        except Exception as e:
            print(f"Error parsing corrected question: {e}")
            print("Raw output:", results)
            return None  # Return None if correction fails

    def sanity_check(
        self,
        question_list: Any,
        topic: str = "Unknown Topic",  # Added to provide context for corrections
        custom_instructions: Optional[str] = None,
        auto_correct: bool = True,  # New parameter to toggle auto-correction
        llm: Optional[Any] = None,
    ) -> SanityCheckSummary:
        """
        Perform a sanity check on a list of questions with optional auto-correction.

        Args:
            question_list (Any): An instance of a question list (MCQList, ShortAnswerQuestionList, etc.).
            topic (str): The topic of the questions, used for corrections.
            custom_instructions (Optional[str]): Additional instructions for the sanity check.
            auto_correct (bool): Whether to auto-correct failed questions (default: True).
            llm (Optional[Any]): Custom LLM instance, defaults to engine's LLM.

        Returns:
            SanityCheckSummary: Summary of sanity check results with corrected questions if applicable.
        """
        if not hasattr(question_list, 'questions') or not question_list.questions:
            raise ValueError("Input must be a valid question list with at least one question")

        llm_to_use = llm if llm is not None else self.llm
        parser = PydanticOutputParser(pydantic_object=SanityCheckSummary)
        format_instructions = parser.get_format_instructions()

        # Determine question type from the first question
        question_type = type(question_list.questions[0]).__name__.replace("Question", "")
        response_model = self._get_response_model(question_type)

        # Prepare the questions for evaluation
        questions_data = []
        for q in question_list.questions:
            options = getattr(q, 'options', None)
            keywords = getattr(q, 'keywords', None)
            answer = str(q.answer) if question_type == "TrueFalse" else q.answer
            questions_data.append({
                "type": question_type,
                "question_text": q.question,
                "answer_text": answer,
                "options": options,
                "keywords": keywords,
                "original_question": q  # Store original for later replacement
            })

        # Define the sanity check prompt
        prompt_template = """
        Perform a sanity check on the following questions and answers.
        For each question, evaluate:
        1. Answer Correctness: Is the answer correct? Respond with 'Correct' or 'Incorrect'.
        2. Question Clarity: Is the question clear and grammatically correct? Respond with 'Clear and Correct' or 'Needs Improvement'.
        3. Distractor Plausibility (if applicable): For Multiple Choice, are distractors plausible but clearly wrong? Respond with 'Plausible Distractors' or 'Implausible Distractors'. For other types, respond with 'N/A'.

        Input questions:
        {questions_input}

        {custom_instructions}

        Output the response in JSON format with a list of results and a summary:
        {format_instructions}
        """
        
        if custom_instructions:
            prompt_template = prompt_template.replace("{custom_instructions}", f"Additional Instructions:\n{custom_instructions}")
        else:
            prompt_template = prompt_template.replace("{custom_instructions}", "")

        # Format the questions input
        questions_input = ""
        for i, q in enumerate(questions_data, 1):
            questions_input += f"\nQuestion {i} ({q['type']}):\n"
            questions_input += f"Question: {q['question_text']}\n"
            questions_input += f"Answer: {q['answer_text']}\n"
            if q['options']:
                options_str = "\n".join(q['options'])
                questions_input += f"Options:\n{options_str}\n"
            if q['keywords']:
                keywords_str = ", ".join(q['keywords'])
                questions_input += f"Keywords: {keywords_str}\n"

        prompt = PromptTemplate(
            input_variables=["questions_input"],
            template=prompt_template,
            partial_variables={"format_instructions": format_instructions}
        )

        # Generate the sanity check results
        sanity_chain = prompt | llm_to_use
        results = sanity_chain.invoke({"questions_input": questions_input})
        results = results.content

        try:
            structured_output = parser.parse(results)
            
            # Adjust results to include options/keywords and calculate pass/fail
            for result, original_q in zip(structured_output.results, questions_data):
                result.options = original_q["options"]
                result.keywords = original_q["keywords"]
                result.passed = (
                    result.answer_correctness == "Correct" and
                    result.question_clarity == "Clear and Correct" and
                    (result.distractor_plausibility in ["Plausible Distractors", "N/A"])
                )

            # Auto-correct failed questions if enabled
            if auto_correct:
                corrected_questions = []
                for i, (result, original_q) in enumerate(zip(structured_output.results, questions_data)):
                    if not result.passed:
                        print(f"Correcting failed question: {result.question}")
                        corrected_q = self._correct_question(result, question_type, topic, llm_to_use)
                        if corrected_q:
                            corrected_questions.append(corrected_q)
                        else:
                            corrected_questions.append(original_q["original_question"])  # Keep original if correction fails
                    else:
                        corrected_questions.append(original_q["original_question"])

                # Update the question list with corrected questions
                question_list.questions = corrected_questions

                # Re-run sanity check on the updated list
                print("Re-running sanity check on corrected questions...")
                return self.sanity_check(question_list, topic, custom_instructions, auto_correct=False, llm=llm_to_use)

            structured_output.total_questions = len(structured_output.results)
            structured_output.passed_questions = sum(1 for r in structured_output.results if r.passed)
            structured_output.failed_questions = structured_output.total_questions - structured_output.passed_questions
            
            return structured_output
        except Exception as e:
            print(f"Error parsing sanity check output: {e}")
            print("Raw output:", results)
            # Return a default summary on error
            default_results = [
                SanityCheckResult(
                    question=q["question_text"],
                    answer=q["answer_text"],
                    options=q["options"],
                    keywords=q["keywords"],
                    answer_correctness="Unknown",
                    question_clarity="Unknown",
                    distractor_plausibility="Unknown",
                    passed=False
                ) for q in questions_data
            ]
            return SanityCheckSummary(
                results=default_results,
                total_questions=len(default_results),
                passed_questions=0,
                failed_questions=len(default_results)
            )
