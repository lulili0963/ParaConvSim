from typing import List, Dict
from src.data_classes.conversational_turn import ConversationalTurn
from .AbstractFeedbackProvider import AbstractFeedbackProvider
from src.simulator.utils import ping_GPT3
import numpy as np
import json


class GPT3FeedbackProvider(AbstractFeedbackProvider):

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


    def give_feedback(self, conversational_turn: ConversationalTurn) -> str:
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
        #politeness: float,
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



        prompt = (f"Your task is to play the role of a user engaging in conversation with the system, getting the information aligned with the information need provided, "
                  f"you should behave as follows: {instructions[c[sample_c[0]]]} Additionally, {instructions[p[sample_p[0]]]} "
                  f"Meanwhile, you should give your feedback to determine whether the system's response aligns with the information need provided. If it doesn't, react accordingly based on your behavior traits. If it does, try to let the system to provide more useful information.\n\n"
                  f"Examples:\n"
                  f"----------\n"
                  f"How a cooperative and polite user behaves:\n\n"
                  f"Information need: The effects of climate changes.\n\n"
                  f"User: I remember Glasgow hosting COP26 last year, but "
                  f"unfortunately I was out of the loop. What was it about?\n"
                  f"System: The COP26 event is a global united Nations summit "
                  f"about climate change and how countries are planning to tackle it."
                  f" The term “climate change” is often used as if it means the same"
                  f" thing as the term “global warming”. The National Academy of "
                  f"Sciences says “climate change” is growing in favor of "
                  f"“global warming” because it helps convey that there are other "
                  f"changes in addition to rising temperatures. In fact, "
                  f"“climate change” means major changes in temperature, rainfall, "
                  f"snow, or wind patterns lasting for decades or longer.\n"
                  f"User: Thanks. I want to know what are the effects of these changes?\n"
                  f"System: Climate change is very likely having an impact now on "
                  f"our planet and its life, according to the latest instalment of "
                  f"a report published by the Intergovernmental Panel on Climate "
                  f"Change (IPCC). And the future problems caused by rising seas, "
                  f"growing deserts, and more frequent droughts all look set to "
                  f"affect the developing world more than rich countries, they add.\n"
                  f"User: I am afraid that is rather vague. Can you be more specific about the effects of climate changes?\n\n"
                  f"----------\n"
                  f"How an uncooperative and polite user behaves:\n\n"
                  f"Information need: The effects of climate changes.\n\n"
                  f"User: I remember Glasgow hosting COP26 last year, but "
                  f"unfortunately I was out of the loop. What was it about?\n"
                  f"System: The COP26 event is a global united Nations summit "
                  f"about climate change and how countries are planning to tackle it."
                  f" The term “climate change” is often used as if it means the same"
                  f" thing as the term “global warming”. The National Academy of "
                  f"Sciences says “climate change” is growing in favor of "
                  f"“global warming” because it helps convey that there are other "
                  f"changes in addition to rising temperatures. In fact, "
                  f"“climate change” means major changes in temperature, rainfall, "
                  f"snow, or wind patterns lasting for decades or longer.\n"
                  f"User: Not relevant but thanks."
                  f"System: Climate change is very likely having an impact now on "
                  f"our planet and its life, according to the latest instalment of "
                  f"a report published by the Intergovernmental Panel on Climate "
                  f"Change (IPCC). And the future problems caused by rising seas, "
                  f"growing deserts, and more frequent droughts all look set to "
                  f"affect the developing world more than rich countries, they add.\n"
                  f"User: I am afraid that is rather vague.\n\n"
                  f"----------\n"
                  f"How a cooperative and impolite user behaves:\n\n"
                  f"Information need: The effects of climate changes.\n\n"
                  f"User: I remember Glasgow hosting COP26 last year, but "
                  f"unfortunately I was out of the loop. What was it about?\n"
                  f"System: The COP26 event is a global united Nations summit "
                  f"about climate change and how countries are planning to tackle it."
                  f" The term “climate change” is often used as if it means the same"
                  f" thing as the term “global warming”. The National Academy of "
                  f"Sciences says “climate change” is growing in favor of "
                  f"“global warming” because it helps convey that there are other "
                  f"changes in addition to rising temperatures. In fact, "
                  f"“climate change” means major changes in temperature, rainfall, "
                  f"snow, or wind patterns lasting for decades or longer.\n"
                  f"User: Whatever. I only want to know what are the effects of these changes?\n"
                  f"System: Climate change is very likely having an impact now on "
                  f"our planet and its life, according to the latest instalment of "
                  f"a report published by the Intergovernmental Panel on Climate "
                  f"Change (IPCC). And the future problems caused by rising seas, "
                  f"growing deserts, and more frequent droughts all look set to "
                  f"affect the developing world more than rich countries, they add.\n"
                  f"User: You are not helpful. That's rather vague and at least can you be more specific about the effects of climate changes?\n\n"
                  f"----------\n"
                  f"How an uncooperative and impolite user behaves:\n\n"
                  f"Information need: The effects of climate changes.\n\n"
                  f"User: I remember Glasgow hosting COP26 last year, but "
                  f"unfortunately I was out of the loop. What was it about?\n"
                  f"System: The COP26 event is a global united Nations summit "
                  f"about climate change and how countries are planning to tackle it."
                  f" The term “climate change” is often used as if it means the same"
                  f" thing as the term “global warming”. The National Academy of "
                  f"Sciences says “climate change” is growing in favor of "
                  f"“global warming” because it helps convey that there are other "
                  f"changes in addition to rising temperatures. In fact, "
                  f"“climate change” means major changes in temperature, rainfall, "
                  f"snow, or wind patterns lasting for decades or longer.\n"
                  f"User: Whatever.\n"
                  f"System: Climate change is very likely having an impact now on "
                  f"our planet and its life, according to the latest instalment of "
                  f"a report published by the Intergovernmental Panel on Climate "
                  f"Change (IPCC). And the future problems caused by rising seas, "
                  f"growing deserts, and more frequent droughts all look set to "
                  f"affect the developing world more than rich countries, they add.\n"
                  f"User: You are not helpful. Too vague.\n\n"
                  f"----------\n"
                  f"Information need: {information_need}\n\n"
                  f"{concatenated_history}"
                  f"User:"
        )
        #cooperativeness -= 0.1
        #print(prompt)
        return prompt, c[sample_c[0]]+'_'+p[sample_p[0]]
