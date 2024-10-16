from abc import abstractmethod

from src.base_module.AbstractModule import AbstractModule
from src.data_classes.conversational_turn import ConversationalTurn


class AbstractAnswerCQ(AbstractModule):
    def __init__(self) -> None:
        """Abstract class for answering clarifying questions."""
        pass

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        response = self.answer_cq(conversational_turn)

        conversational_turn.update_history(response, "User", utterance_type="answer")
        return conversational_turn

    @abstractmethod
    def answer_cq(self, conversational_turn: ConversationalTurn) -> str:
        """Answers given clarifying question based on self.information_need.

        Args:
            conversational_turn: Object containing all information about a turn.
        """
        raise NotImplementedError
