from .load_data import load_vector_db

class Retrieve:
    def __init__(self) -> None:
        self.data = load_vector_db()
        
    def semantic_search(self, query):
        content_string = ""
        
        results = self.data.similarity_search_with_score(
            query,
            k=2,
        )
        for i, (res, score) in enumerate(results):
            if i > 0:
                content_string += "\n"  # Add newline between paragraphs
            content_string += res.page_content
        
        return content_string