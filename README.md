# ParaConvSim
Code for the SIGIR-AP 2024 paper "Simulating Conversational Search Users with Parameterized Behavior".


![system](https://github.com/user-attachments/assets/16a4c961-a859-4173-b8ce-c24f55db46c2)

We propose ParaConvSim to simulate users with three different behavior traits: Patience, Cooperativeness and Politeness, achieving personalized feedback. 
Then, we evaluate the retrieval performance on TREC CAsT Year4 from [1].

Note: the framework of the pipeline is from SIGIR 2023 paper "Exploiting Simulated User Feedback for Conversational Search: Ranking, Rewriting, and Beyond".

## How to Run

Clone the repository.

Install all necessary libraries

```
pip install -r requirements.txt
```

Download CAsT benchmark and other necessary artifacts like a subset of the indexed document collection.

```
bash setup.sh
```

Then set your openai API key. It should look something like:
```
# API Key for GPT3
OPENAI_API_KEY="api_key"
```

Run. configs file contain all the utilized parameters to controll the level of cooperativeness and politness.

```
python main.py
```

If you want to customize your own parameters, pls add a new json into the configs file.

If you want to adjust patience, you can go to src/base_module/Pipelines.py, set something like:


```

```














## Reference
[1] Owoicho, P., Sekulic, I., Aliannejadi, M., Dalton, J., & Crestani, F. (2023, July). Exploiting simulated user feedback for conversational search: Ranking, rewriting, and beyond. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 632-642).
