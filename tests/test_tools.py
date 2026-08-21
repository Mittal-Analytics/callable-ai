from inspect import signature

from mittal_ai import format_docstring, partial_with_doc


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
