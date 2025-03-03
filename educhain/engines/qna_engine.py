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
import time
import csv

QuestionType = Literal["Multiple Choice", "Short Answer", "True/False", "Fill in the Blank"]

class QnAEngine:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig()
        self.llm = self._initialize_llm(llm_config)

    def _initialize_llm(self, llm_config: LLMConfig):
        """Initialize the language model based on the provided configuration."""
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

    # Question Generation with Automated Retries
    def generate_questions(
        self,
        topic: str,
        num: int = 1,
        question_type: QuestionType = "Multiple Choice",
        prompt_template: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        response_model: Optional[Type[Any]] = None,
        batch_size: int = 5,
        max_retries_per_batch: int = 3,
        **kwargs
    ) -> Any:
        """
        Generate the requested number of questions in batches with automated retries.
        """
        response_model = response_model or self._get_response_model(question_type)
        accumulated_questions = []

        while len(accumulated_questions) < num:
            remaining = num - len(accumulated_questions)
            current_batch_size = min(batch_size, remaining)

            batch_success = False
            for attempt in range(max_retries_per_batch):
                try:
                    batch_result = self._generate_batch(
                        topic=topic,
                        num=current_batch_size,
                        question_type=question_type,
                        prompt_template=prompt_template,
                        custom_instructions=custom_instructions,
                        response_model=response_model,
                        **kwargs
                    )
                    if batch_result and hasattr(batch_result, 'questions') and batch_result.questions:
                        accumulated_questions.extend(batch_result.questions)
                        print(f"Batch attempt {attempt + 1}: Generated {len(batch_result.questions)} questions")
                        batch_success = True
                        break
                    else:
                        raise ValueError("Batch generated no questions")
                except Exception as e:
                    print(f"Batch attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries_per_batch - 1:
                        print(f"Retrying batch... ({attempt + 2}/{max_retries_per_batch})")
                        time.sleep(1)  # Backoff
                    else:
                        print(f"Max retries ({max_retries_per_batch}) reached for this batch.")

            if not batch_success and current_batch_size > 1:
                print(f"Reducing batch size from {current_batch_size} to {current_batch_size - 1}")
                batch_size = max(1, current_batch_size - 1)
            elif not batch_success:
                print("Even batch size of 1 failed. Continuing with partial results.")
                break

        if len(accumulated_questions) > num:
            accumulated_questions = accumulated_questions[:num]

        print(f"Total generated: {len(accumulated_questions)} out of {num} requested")
        return response_model(questions=accumulated_questions)

    def _generate_batch(
        self,
        topic: str,
        num: int,
        question_type: QuestionType,
        prompt_template: Optional[str],
        custom_instructions: Optional[str],
        response_model: Type[Any],
        **kwargs
    ) -> Any:
        """Generate a single batch of questions."""
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
            print(f"Error parsing batch output: {e}")
            print("Raw output:", results)
            raise

    # Load Custom Questions from CSV
    def load_questions_from_csv(
        self,
        csv_path: str,
        question_type: QuestionType = "Multiple Choice",
        question_col: str = "Question Text",
        answer_col: str = "Correct Answer",
        options_cols: Optional[list] = None,
        explanation_col: Optional[str] = "Explanation"
    ) -> Any:
        """
        Load questions from a CSV file and convert them into a compatible question list format.
        """
        response_model = self._get_response_model(question_type)
        single_response_model = self._get_single_response_model(question_type)
        questions = []

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    if question_type == "Multiple Choice":
                        options = [row[col] for col in options_cols] if options_cols else []
                        question = single_response_model(
                            question=row[question_col],
                            answer=row[answer_col],
                            options=options,
                            explanation=row.get(explanation_col)
                        )
                    elif question_type == "Short Answer":
                        question = single_response_model(
                            question=row[question_col],
                            answer=row[answer_col],
                            keywords=row.get(explanation_col, "").split(", ") if explanation_col else [],
                            explanation=row.get(explanation_col)
                        )
                    elif question_type == "True/False":
                        question = single_response_model(
                            question=row[question_col],
                            answer=row[answer_col].lower() in ["true", "t", "1"],
                            explanation=row.get(explanation_col)
                        )
                    elif question_type == "Fill in the Blank":
                        question = single_response_model(
                            question=row[question_col],
                            answer=row[answer_col],
                            blank_word=row[answer_col],
                            explanation=row.get(explanation_col)
                        )
                    questions.append(question)
                except KeyError as e:
                    print(f"Skipping row due to missing column {e}: {row}")
                except Exception as e:
                    print(f"Error processing row: {e} - {row}")

        return response_model(questions=questions)

    # Correct Failed Questions
    def _correct_question(
        self,
        failed_result: SanityCheckResult,
        question_type: QuestionType,
        topic: str,
        llm: Optional[Any] = None
    ) -> Any:
        """Correct a failed question with retries."""
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

        max_retries = 3
        for attempt in range(max_retries):
            try:
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
                corrected_question = parser.parse(results)
                return corrected_question
            except Exception as e:
                print(f"Correction attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying correction... ({attempt + 2}/{max_retries})")
                    time.sleep(1)
                else:
                    print("Max retries reached for correction. Returning None.")
                    return None

    # Sanity Check with Retries (for both Educhain-generated and custom questions)
    def sanity_check(
        self,
        question_list: Any,
        topic: str = "Unknown Topic",
        custom_instructions: Optional[str] = None,
        auto_correct: bool = True,
        llm: Optional[Any] = None,
        batch_size: int = 10,
        max_retries_per_batch: int = 3
    ) -> SanityCheckSummary:
        """
        Perform a sanity check on a list of questions with retries and optional auto-correction.
        Updates question_list in-place if auto_correct is True.
        """
        if not hasattr(question_list, 'questions') or not question_list.questions:
            raise ValueError("Input must be a valid question list with at least one question")

        llm_to_use = llm if llm is not None else self.llm
        parser = PydanticOutputParser(pydantic_object=SanityCheckSummary)
        format_instructions = parser.get_format_instructions()

        question_type = type(question_list.questions[0]).__name__.replace("Question", "")
        response_model = self._get_response_model(question_type)

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
                "original_question": q
            })

        all_results = []
        for i in range(0, len(questions_data), batch_size):
            batch_questions = questions_data[i:i + batch_size]
            print(f"Processing sanity check batch {i // batch_size + 1} ({len(batch_questions)} questions)")

            prompt_template = """
            Perform a sanity check on the following questions about {topic}.
            For each question:
            1. Answer Correctness: Verify if the provided answer is mathematically correct. Respond with 'Correct' or 'Incorrect'.
            2. Question Clarity: Is the question clear and grammatically correct? Respond with 'Clear and Correct' or 'Needs Improvement'.
            3. Distractor Plausibility (if applicable): For Multiple Choice, are distractors plausible but clearly wrong? Respond with 'Plausible Distractors' or 'Implausible Distractors'. For other types, respond with 'N/A'.

            When checking arithmetic (e.g., fractions):
            - For addition/subtraction, ensure common denominators are used correctly.
            - For multiplication, multiply numerators and denominators directly.
            - For division, multiply by the reciprocal.
            - Simplify fractions and compare with the provided answer.

            Input questions:
            {questions_input}

            {custom_instructions}

            Output the response in JSON format with a list of results:
            {format_instructions}
            """

            if custom_instructions:
                prompt_template = prompt_template.replace("{custom_instructions}", f"Additional Instructions:\n{custom_instructions}")
            else:
                prompt_template = prompt_template.replace("{custom_instructions}", "")

            questions_input = ""
            for j, q in enumerate(batch_questions, i + 1):
                questions_input += f"\nQuestion {j} ({q['type']}):\n"
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
                partial_variables={"format_instructions": format_instructions, "topic": topic}
            )

            batch_success = False
            for attempt in range(max_retries_per_batch):
                try:
                    sanity_chain = prompt | llm_to_use
                    results = sanity_chain.invoke({"questions_input": questions_input})
                    results = results.content
                    batch_output = parser.parse(results)
                    all_results.extend(batch_output.results)
                    batch_success = True
                    break
                except Exception as e:
                    print(f"Sanity check attempt {attempt + 1} failed for batch {i // batch_size + 1}: {e}")
                    if attempt < max_retries_per_batch - 1:
                        print(f"Retrying batch... ({attempt + 2}/{max_retries_per_batch})")
                        time.sleep(1)
                    else:
                        print(f"Max retries ({max_retries_per_batch}) reached for this batch.")
                        batch_results = [
                            SanityCheckResult(
                                question=q["question_text"],
                                answer=q["answer_text"],
                                options=q["options"],
                                keywords=q["keywords"],
                                answer_correctness="Unknown",
                                question_clarity="Unknown",
                                distractor_plausibility="Unknown",
                                passed=False
                            ) for q in batch_questions
                        ]
                        all_results.extend(batch_results)

            if not batch_success:
                print("Batch failed after retries. Continuing with partial results.")

        structured_output = SanityCheckSummary(
            results=all_results,
            total_questions=len(all_results),
            passed_questions=0,
            failed_questions=0
        )

        for result, original_q in zip(structured_output.results, questions_data):
            result.options = original_q["options"]
            result.keywords = original_q["keywords"]
            result.passed = (
                result.answer_correctness == "Correct" and
                result.question_clarity == "Clear and Correct" and
                (result.distractor_plausibility in ["Plausible Distractors", "N/A"])
            )

        if auto_correct:
            corrected_questions = []
            for i, (result, original_q) in enumerate(zip(structured_output.results, questions_data)):
                if not result.passed:
                    print(f"Correcting failed question {i + 1}: {result.question}")
                    corrected_q = self._correct_question(result, question_type, topic, llm_to_use)
                    if corrected_q:
                        corrected_questions.append(corrected_q)
                    else:
                        corrected_questions.append(original_q["original_question"])
                else:
                    corrected_questions.append(original_q["original_question"])

            question_list.questions = corrected_questions
            print("Re-running sanity check on corrected questions...")
            return self.sanity_check(question_list, topic, custom_instructions, auto_correct=False, llm=llm_to_use)

        structured_output.total_questions = len(structured_output.results)
        structured_output.passed_questions = sum(1 for r in structured_output.results if r.passed)
        structured_output.failed_questions = structured_output.total_questions - structured_output.passed_questions
        
     
