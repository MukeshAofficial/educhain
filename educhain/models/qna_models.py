from educhain.models.base_models import BaseQuestion, QuestionList
from pydantic import BaseModel, Field
from typing import List, Optional

class MultipleChoiceQuestion(BaseQuestion):
    options: List[str]

    def show(self):
        print(f"Question: {self.question}")
        options_str = "\n".join(f"  {chr(65 + i)}. {option}" for i, option in enumerate(self.options))
        print(f"Options:\n{options_str}")
        print(f"\nCorrect Answer: {self.answer}")
        if self.explanation:
            print(f"Explanation: {self.explanation}")
        print()

class ShortAnswerQuestion(BaseQuestion):
    keywords: List[str] = Field(default_factory=list)

    def show(self):
        super().show()
        if self.keywords:
            print(f"Keywords: {', '.join(self.keywords)}")
        print()

class TrueFalseQuestion(BaseQuestion):
    answer: bool

    def show(self):
        super().show()
        print(f"True/False: {self.answer}")
        print()

class FillInBlankQuestion(BaseQuestion):
    blank_word: Optional[str] = None

    def show(self):
        super().show()
        print(f"Word to fill: {self.blank_word or self.answer}")
        print()

class MCQList(QuestionList):
    questions: List[MultipleChoiceQuestion]

class ShortAnswerQuestionList(QuestionList):
    questions: List[ShortAnswerQuestion]

class TrueFalseQuestionList(QuestionList):
    questions: List[TrueFalseQuestion]

class FillInBlankQuestionList(QuestionList):
    questions: List[FillInBlankQuestion]

class SanityCheckResult(BaseModel):
    question: str = Field(..., description="The question being evaluated")
    answer: str = Field(..., description="The correct answer")
    options: Optional[List[str]] = Field(None, description="Options if MCQ, else None")
    keywords: Optional[List[str]] = Field(None, description="Keywords if Short Answer, else None")
    answer_correctness: str = Field(..., description="Correct or Incorrect")
    question_clarity: str = Field(..., description="Clear and Correct or Needs Improvement")
    distractor_plausibility: str = Field(..., description="Plausible Distractors, Implausible Distractors, or N/A")
    passed: bool = Field(..., description="Whether the question passed the sanity check")

class SanityCheckSummary(BaseModel):
    results: List[SanityCheckResult] = Field(..., description="List of individual sanity check results")
    total_questions: int = Field(..., description="Total number of questions checked")
    passed_questions: int = Field(..., description="Number of questions that passed")
    failed_questions: int = Field(..., description="Number of questions that failed")

    def show(self):
        print("=" * 80)
        print(f"Sanity Check Summary")
        print(f"Total Questions: {self.total_questions}")
        print(f"Passed: {self.passed_questions} | Failed: {self.failed_questions}")
        print("=" * 80)
        for i, result in enumerate(self.results, 1):
            print(f"\nQuestion {i}: {result.question}")
            print(f"Answer: {result.answer}")
            if result.options:
                print("Options:")
                for j, opt in enumerate(result.options):
                    print(f"  {chr(65+j)}. {opt}")
            if result.keywords:
                print(f"Keywords: {', '.join(result.keywords)}")
            print(f"Answer Correctness: {result.answer_correctness}")
            print(f"Question Clarity: {result.question_clarity}")
            print(f"Distractor Plausibility: {result.distractor_plausibility}")
            print(f"Result: {'Passed' if result.passed else 'Failed'}")
        print("=" * 80)
