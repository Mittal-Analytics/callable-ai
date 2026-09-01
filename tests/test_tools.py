from enum import Enum
from inspect import signature

from callable_ai import format_docstring, partial_with_doc
from callable_ai.responses import _parse_docs


def test_format_docstring():
    @format_docstring(item="document")
    def tool():
        """Read the {item}."""

    assert tool.__doc__ == "Read the document."


def test_partial_with_doc_preserves_tool_metadata_and_hides_bound_arguments():
    def tool(company_id: int, question: str) -> str:
        """Answer a company question."""
        return f"{company_id}: {question}"

    wrapped = partial_with_doc(tool, company_id=123)

    assert wrapped(question="hello") == "123: hello"
    assert getattr(wrapped, "__name__") == "tool"
    assert wrapped.__doc__ == tool.__doc__
    assert list(signature(wrapped).parameters) == ["question"]


def test_parse_docs_preserves_nested_argument_details():
    class ApproachEnum(Enum):
        SEARCH_AND_ANSWER = "search_and_answer"
        READ_FULL_ANNUAL_REPORT_TO_ANSWER = "read_full_annual_report_to_answer"
        NOT_SURE = "not_sure"

    @format_docstring(
        SEARCH_AND_ANSWER=ApproachEnum.SEARCH_AND_ANSWER.value,
        READ_FULL_ANNUAL_REPORT_TO_ANSWER=(
            ApproachEnum.READ_FULL_ANNUAL_REPORT_TO_ANSWER.value
        ),
        NOT_SURE=ApproachEnum.NOT_SURE.value,
    )
    async def refer_annual_report(
        company_id: int, document_id: int, approach: ApproachEnum, question: str
    ) -> None:
        """
        Refer the Annual Report (or Red Herring Prospectus) for specific details or questions.

        Args:
            - company_id: The ID of the company the document should belong to.
            - document_id: The ID of the annual report/RHP.
            - approach: Select the appropriate approach based on the query:
                - {SEARCH_AND_ANSWER}: The sub-agent does a keyword search in the annual report to get the answer from limited search result pages. Use this when you know exactly what you are looking for. It is fast and cheap.
                - {READ_FULL_ANNUAL_REPORT_TO_ANSWER}: Best for summaries, subjective queries, or broad topics. The sub-agent reads the full document to get the answer.
                - {NOT_SURE}: Use this if you are not sure which method to use.
            - question: The question or specific data point you are looking for.
        """

    parsed = _parse_docs(refer_annual_report.__doc__)
    arguments = parsed["arguments"]

    assert parsed["function_docs"] == (
        "Refer the Annual Report (or Red Herring Prospectus) for specific "
        "details or questions."
    )
    assert set(arguments) == {"company_id", "document_id", "approach", "question"}
    assert arguments["company_id"] == (
        "The ID of the company the document should belong to."
    )
    assert arguments["document_id"] == "The ID of the annual report/RHP."
    assert arguments["question"] == (
        "The question or specific data point you are looking for."
    )
    assert arguments["approach"].splitlines() == [
        "Select the appropriate approach based on the query:",
        "- search_and_answer: The sub-agent does a keyword search in the annual "
        "report to get the answer from limited search result pages. Use this when "
        "you know exactly what you are looking for. It is fast and cheap.",
        "- read_full_annual_report_to_answer: Best for summaries, subjective "
        "queries, or broad topics. The sub-agent reads the full document to get "
        "the answer.",
        "- not_sure: Use this if you are not sure which method to use.",
    ]
