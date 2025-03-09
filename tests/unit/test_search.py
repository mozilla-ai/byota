import numpy as np
from byota.search import SearchService
from scipy.spatial import distance_matrix
from loguru import logger


def test_search():
    embeddings = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 3],
            [1, 1, 0, 0],
        ]
    ).astype(np.float32)

    # as our query will be and index, we do not need an embedding service so we pass None
    search_service = SearchService(embeddings, None)
    indices = search_service.most_similar_indices(0, 8)

    # log distance matrix and nearest neighbor indices
    logger.debug(distance_matrix(embeddings, embeddings))
    logger.debug(indices)

    assert list(indices) == [0, 4, 1, 2, 3, 5]
