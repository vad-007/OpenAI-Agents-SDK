import unittest
from pathlib import Path

from rag import DocumentChunk, load_documents, retrieve_documents


class RetrieveDocumentsTests(unittest.TestCase):
    def test_loads_markdown_chunks_with_source_labels(self):
        chunks = load_documents(Path(__file__).resolve().parents[1] / "knowledge_base")

        self.assertTrue(chunks)
        self.assertEqual(chunks[0].citation, "agents_sdk_basics.md#chunk-1")

    def test_ranks_passages_by_matching_query_terms(self):
        chunks = [
            DocumentChunk(
                "guide.md", 1,
                "An agent uses a function tool to execute an application function.",
            ),
            DocumentChunk("guide.md", 2, "A handoff transfers the request to a specialist agent."),
        ]

        results = retrieve_documents("How does a handoff transfer to a specialist agent?", chunks)

        self.assertEqual([chunk.chunk_number for chunk in results], [2, 1])

    def test_returns_no_results_for_unmatched_or_empty_search(self):
        chunks = [DocumentChunk("guide.md", 1, "An agent uses instructions and tools.")]

        self.assertEqual(retrieve_documents("volcanic eruption", chunks), [])
        self.assertEqual(retrieve_documents("the and how", chunks), [])

    def test_limits_number_of_results(self):
        chunks = [
            DocumentChunk("guide.md", index, "Agent tools let an agent call functions.")
            for index in range(1, 4)
        ]

        self.assertEqual(len(retrieve_documents("agent tools", chunks, limit=2)), 2)


if __name__ == "__main__":
    unittest.main()
