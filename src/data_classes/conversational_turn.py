from dataclasses import dataclass, field
from typing import Dict, List

from ir_measures import Qrel, parse_measure, ScoredDoc


@dataclass
class Document:
    doc_id: str
    doc_text: str
    score: float = None


@dataclass
class ConversationalTurn:
    turn_id: str  # ideally should be an int, but CAsT year 4 is a string
    information_need: str
    user_utterance: str
    personalizedpara: str #update sampling parameters
    #cooperativeness: float  #add cooperativeness parameters    
    #politeness: float #add politeness parameters
    # One of: ["question", "feedback", "comment", "answer"]
    user_utterance_type: str
    relevance_judgements: List[Qrel]
    rewritten_utterance: str = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    ranking: List[Document] = None
    system_response: str = None
    # One of: ["clarifying_question", "response"]
    system_response_type: str = None
    feedback_rounds: int = 0
    

    


    def update_history(
        self,
        utterance: str = None,
        participant: str = None,
        utterance_type: str = None,
        ranking: List[Document] = None,
    ) -> None:
        """Update conversational history and current turn info.

        Args:
            utterance: new utterance.
            participant: "System" or "User".
            utterance_type: only valid when participant is "System".
            ranking: a new ranking of documents to be updated.
        Returns:
            None.
        """

        # it doesn't store initial query to history
        if participant == "User":
            self.conversation_history += [
                {
                    "participant": "User",
                    "utterance": self.user_utterance,
                    "utterance_type": self.user_utterance_type,
                    "rewritten_utterance": self.rewritten_utterance,
                    "parameters": self.personalizedpara #Update personalized parameters
                }
            ]
            # Since the result is in the format of utterance+ para: +parameters
            # Split the result to get pure utterance
            self.user_utterance = utterance.split(' para:')[0] #utterance
            self.user_utterance_type = utterance_type
            self.rewritten_utterance = None
            # Split the result to get parameters
            self.personalizedpara = utterance.split(' para:')[1] if len(utterance.split(' para:')) >1 else None #parameters

        elif participant == "System":
            if self.system_response:
                self.conversation_history += [
                    {
                        "participant": "System", "utterance": self.system_response,
                        "utterance_type": self.system_response_type,
                    }
                ]
            self.system_response = utterance
            self.system_response_type = utterance_type

            if ranking:
                self.ranking = ranking

    def evaluate_turn(self, measure: str = 'nDCG@3'):
        """Evaluates the ranking in this class based on relevance judgements.

        Args:
            metric: string representation of measure of interest
        Returns:
            Calculated measure: float
        """

        documents = [
            ScoredDoc(self.turn_id, document.doc_id, document.score) for
            document in self.ranking
        ]
        parsed_measure = parse_measure(measure)
        score = parsed_measure.calc_aggregate(
            self.relevance_judgements, documents
        )

        return score

