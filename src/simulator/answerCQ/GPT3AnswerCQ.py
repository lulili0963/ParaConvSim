from typing import List, Dict
from src.data_classes.conversational_turn import ConversationalTurn
from .AbstractAnswerCQ import AbstractAnswerCQ
from src.simulator.utils import ping_GPT3
import numpy as np
import json


class GPT3AnswerCQ(AbstractAnswerCQ):

    def __init__(
        self, parameters_path: str, instructions_path: str) -> None:
        # Instructions and parameters loading
        self.parameters = self.load_json_file(parameters_path)
        self.instructions = self.load_json_file(instructions_path)
    #Load json file
    def load_json_file(self, path: str) -> dict:
        with open(path, 'r') as json_file:
              results = json.load(json_file)
        return results

    

    def answer_cq(self, conversational_turn: ConversationalTurn) -> str:
        prompt, parameters = self.create_prompt(
            conversational_turn.information_need, 
            conversational_turn.conversation_history, 
            conversational_turn.user_utterance, 
            conversational_turn.system_response,
            self.parameters,
            self.instructions
        )

        return ping_GPT3(prompt)+' para:'+parameters

    @staticmethod
    def create_prompt(
        information_need: str,
        history: List[str],
        current_user_turn: str,
        current_system_response: str,
        parameters: Dict,
        instructions: Dict
        #cooperativeness: float,
        #politeness: float
    ) -> str:
        concatenated_history = ""
        for turn in history:
            if turn["participant"] == "User":
                concatenated_history += f'User: {turn["utterance"]}\n'
            elif turn["participant"] == "System":
                concatenated_history += f'System: {turn["utterance"]}\n'

        concatenated_history += f"User: {current_user_turn}\n"
        concatenated_history += f"System: {current_system_response}\n"

        #Zero-shot and Inject the parameters: cooperativeness/politeness into the prompt
        # Inject the cooperativenss/politeness into the prompt
        c = {1: 'cooperative', 0: 'uncoop'}
        p = {1: 'polite', 0: 'imp'}
        


        # Sample parameters: cooperativeness/politeness from bernoulli distribution
        sample_c = np.random.binomial(size=1, n=1, p=parameters['cooperativeness'])
        print('sample from cooperativeness:',sample_c[0],c[sample_c[0]])
        sample_p = np.random.binomial(size=1, n=1, p=parameters['politeness'])
        print('sample from politeness:',sample_p[0],p[sample_p[0]])

        

        prompt = ("As a user, you are interacting with the system in a conversation. "+instructions[c[sample_c[0]]]+"And also "+instructions[p[sample_p[0]]]+\
             "Generate a response to the system question based on the "
             "conversation and information needs, and the response should be aligned with your personality:\n\n"
            f"Information need: {information_need}\n\n"
            f"{concatenated_history}"
            "User:"
        )
        #cooperativeness -= 0.1

        return prompt, c[sample_c[0]]+'_'+p[sample_p[0]]

