from .AbstractModule import AbstractModule
from src.simulator.provide_feedback import AbstractFeedbackProvider

from typing import List
from src.data_classes.conversational_turn import ConversationalTurn
import numpy as np

class Pipeline(AbstractModule):
    """Single pass through all conversational modules"""

    def __init__(self, modules: List[AbstractModule]) -> None:
        self.modules = modules

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        for module in self.modules:
            conversational_turn = module.step(conversational_turn)

        return conversational_turn


class RecursivePipeline(Pipeline):
    """Allows for multiple feedback rounds"""


    """
      Add two parameters:
      cooperativeness : sample from bernoulli distribution. 
                        The probability of cooperativeness is configured in the parameters.json file.
                       

      politeness : sample from bernoulli distribution. 
                   The probability of cooperativeness is configured in the parameters.json file.

    """

    def __init__(self, modules: List[AbstractModule], max_feedback_rounds=1, min_ndcg=1.0) -> None:
        super().__init__(modules)
        self.max_feedback_rounds = max_feedback_rounds
        self.min_ndcg = min_ndcg

    def step(self, conversational_turn: ConversationalTurn) -> ConversationalTurn:
        for module in self.modules:
            conversational_turn = module.step(conversational_turn)
            if isinstance(module, AbstractFeedbackProvider) and \
                conversational_turn.feedback_rounds < self.max_feedback_rounds and \
                    conversational_turn.evaluate_turn() < self.min_ndcg:
                #with open(f"data/f_moreround_fewshot_combinedex/para_generated_conversations/semantic_cq_{conversational_turn.feedback_rounds}.run", "a") as f:
                  #for index, document in enumerate(conversational_turn.ranking):
                    #f.write(f"{conversational_turn.turn_id}\tQ0\t{document.doc_id}\t{index+1}\t{document.score}\tsemantic_cq\n")
                conversational_turn.feedback_rounds += 1
                conversational_turn = self.step(conversational_turn)

        return conversational_turn
