from abc import abstractmethod
from typing import List

from src.data_classes.conversational_turn import ConversationalTurn, Document
from src.base_module.AbstractModule import AbstractModule


class AbstractReranker(AbstractModule):
    def __init__(self):
        """Abstract class for reranking."""
        pass

    @abstractmethod
    def rerank(self, conversational_turn: ConversationalTurn) -> List[Document]:
        """Method for initial retrieval.

        Args:
            conversational_turn: A class representing conversational turn,
                containing query, history, clarfifying question, answer.
        Raises:
            NotImplementedError: Raised if the method is not implemented.
        Returns:
            Re-ranked list of documents.
        """
        raise NotImplementedError

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        ranking = self.rerank(conversational_turn)
        conversational_turn.ranking = ranking
        return conversational_turn
