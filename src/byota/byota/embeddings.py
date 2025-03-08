import json
import numpy as np
import requests

# -- Embeddings --------------------------------------------------------------


class EmbeddingService:
    def __init__(self, url: str, model: str = None):
        self._url = url
        self._model = model

    def is_working(self) -> bool:
        """Checks if the service is there and working by trying
        to send an actual embedding request.
        """
        pass

    def get_embedding(self, text: str) -> list:
        """Given an input text, returns the embeddings as calculated
        by the embedding service.
        """
        pass

    def calculate_embeddings(self, texts: list[str], bar=None) -> np.ndarray:
        """Given a list of input texts, returns all the embeddings
        as a numpy array.
        """

        embeddings = []
        for i, t in enumerate(texts):
            embeddings.append(self.get_embedding(str(t)))
            if bar is not None:
                bar.update()
            if not (i % 10):
                print(".", end="")
        return np.array(embeddings)


class LLamafileEmbeddingService(EmbeddingService):
    def is_working(self):
        response = requests.request(
            url=self._url,
            method="POST",
        )
        return response.status_code == 200

    def get_embedding(self, text: str) -> list:
        try:
            response = requests.request(
                url=self._url,
                method="POST",
                data={"content": text},
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            raise

        return json.loads(response.text)["embedding"]


class OllamaEmbeddingService(EmbeddingService):
    def __init__(self, url: str, model: str):
        # model is compulsory for ollama
        super().__init__(url, model)

    def is_working(self):
        response = requests.request(
            url=self._url,
            method="POST",
            data=json.dumps({"model": self._model, "input": ""}),
        )
        return response.status_code

    def get_embedding(self, text: str):
        # workaround for ollama breaking with empty input text
        if not text:
            text = " "

        try:
            response = requests.request(
                url=self._url,
                method="POST",
                data=json.dumps({"model": self._model, "input": text}),
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            raise

        return json.loads(response.text)["embeddings"][0]
