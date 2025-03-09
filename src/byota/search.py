from scipy import spatial
import numpy as np
from byota.embeddings import EmbeddingService
from loguru import logger

# -- Similarity --------------------------------------------------------------


class SearchService:
    def __init__(self, embeddings: np.ndarray, embedding_service: EmbeddingService):
        self._embeddings = embeddings
        self._embedding_service = embedding_service
        self._tree = spatial.KDTree(self._embeddings)

    def prepare_query(self, query):
        """A query can either be an integer ID (index in the dataframe)
        or a string. As similarity is calculated among embeddings, this
        method makes sure we always return an embedding.
        """

        def is_integer_string(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        if is_integer_string(query):
            return self._embeddings[int(query)]
        else:
            return self._embedding_service.get_embedding(query)

    def most_similar_indices(self, query, k=5):
        """Given a query (whether as an integer index to a status or plain
        text), return the k indices of the most similar embeddings.
        """
        if k>len(self._embeddings):
            logger.warning("The number of neighbors k is greater than the number of samples. Setting k=num_samples")
            k=len(self._embeddings)

        q = self.prepare_query(query)

        # get the k nearest neighbors' indices
        return self._tree.query(q, k=k + 1)[1]

    def most_similar_embeddings(self, query, k=5):
        """Given a query (whether as an integer index to a status or plain
        text), return the k most similar embeddings."""
        indices = self.most_similar_indices(query, k)

        return self._embeddings[indices]
