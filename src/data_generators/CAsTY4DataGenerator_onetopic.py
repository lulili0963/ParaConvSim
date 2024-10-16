"""
    ATTENTION:
    Temporary file
    Only for testing codes, testing all the codes on one topic -number 132
    
"""
from .AbstractConversationalDataGenerator import (
    AbstractConversationalDataGenerator
)
import json
from ir_measures import read_trec_qrels
from src.data_classes import ConversationalTurn


class CAsTY4DataGenerator(AbstractConversationalDataGenerator):

    def __init__(self, dataset_path: str, relevance_judgements_path: str) :
        with open(dataset_path) as cast_y4_topics_file:
            self.topics = json.load(cast_y4_topics_file)
            print(self.topics[0])

        self.qrels = list(read_trec_qrels(relevance_judgements_path))

    def get_turn(self) -> ConversationalTurn:
        parsed_turns = set()
        # Only for initail experiments and only test on the first topic - number132
        for topic in [self.topics[0]]:
            for index, turn in enumerate(topic['turn']):
                #print(index)
                turn_id = f"{topic['number']}_{turn['number']}"
                if turn_id in parsed_turns:
                    continue
                information_need = turn.get("information_need")
                utterance = turn.get("utterance").split(' para:')[0]
                para = turn.get("utterance").split(' para:')[1] if len(turn.get("utterance").split(' para:')) > 1 else None
                utterance_type = turn.get("utterance_type").lower()
                relevance_judgements = [
                    qrel for qrel in self.qrels if qrel.query_id == turn_id]
                conversational_history = []
                for previous_turn in topic['turn'][:index]:
                    # extract utterance attributes
                    previous_user_utterance = previous_turn.get("utterance").split(' para:')[0]
                    previous_para = previous_turn.get("utterance").split(' para:')[1] if len(previous_turn.get("utterance").split(' para:')) > 1 else None
                    previous_user_utterance_rewrite = previous_turn.get(
                        "automatic_rewritten_utterance")
                    previous_user_utterance_type = previous_turn.get(
                        "utterance_type")

                    # extract response attributes
                    previous_system_response = previous_turn.get("response")
                    previous_system_response_type = previous_turn.get(
                        "response_type")

                    conversational_history += [
                        {
                            "participant": "User", 
                            "utterance": previous_user_utterance,
                            "utterance_type": previous_user_utterance_type, 
                            "rewritten_utterance": previous_user_utterance_rewrite,
                            "parameters": previous_para
                            
                        },
                        {
                            "participant": "System", 
                            "utterance": previous_system_response,
                            "utterance_type": previous_system_response_type
                        }
                    ]
                parsed_turns.add(turn_id)
                yield ConversationalTurn(
                    turn_id=turn_id, information_need=information_need,
                    user_utterance=utterance,
                    conversation_history=conversational_history,
                    user_utterance_type=utterance_type,
                    relevance_judgements=relevance_judgements,
                    personalizedpara=para

                )
