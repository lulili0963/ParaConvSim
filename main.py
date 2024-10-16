from src.base_module.Pipelines import RecursivePipeline
from src.data_generators import CAsTY4DataGenerator
from src.simulator.provide_feedback import GPT3FeedbackProvider
from src.mi_systems.retriever import SparseRetriever
from src.mi_systems.reranker import T5Ranker
from src.mi_systems.rewriter import T5Rewriter
from src.mi_systems.response_generator import BARTResponseGenerator

import json
import os
from pathlib import Path
from tqdm import tqdm


for p in os.listdir("4extremerepeated"):
    if os.path.splitext(p)[1] == ".json":
        print(p)
        print(f"4extremerepeated/{p}")
        run_name = "semantic_cq"
        base_path = f"data/f_moreround_fewshot_combinedex_round1_4extremerepeated/para_generated_conversations_{os.path.splitext(p)[0]}"  # noqa
        output_path = f"{base_path}/{run_name}/transcripts"

        data_generator = CAsTY4DataGenerator(
            dataset_path="data/cast/year_4/annotated_topics.json",
            relevance_judgements_path="data/cast/year_4/cast2022.qrel",
        )

        pipeline = RecursivePipeline(
            [
                T5Rewriter(),
                SparseRetriever(
                    collection="data/cast/year_4/indexes/content/files/index/sparse/",  # noqa
                    collection_type="json",
                ),
                T5Ranker(),
                BARTResponseGenerator(),
                GPT3FeedbackProvider(f"4extremerepeated/{p}", "instructions.json"),  # noqa
            ]
        )

        """
    pipeline = Pipeline([
        T5Rewriter(),
        SparseRetriever(
            collection="data/cast/year_4/indexes/content/files/index/sparse/",
            collection_type="json"),
        T5Ranker(),
        BARTResponseGenerator(),
        GPT3FeedbackProvider("parameters.json", "instructions.json"),
        T5Rewriter(),
        SparseRetriever(
            collection="data/cast/year_4/indexes/content/files/index/sparse/",
            collection_type="json"),
        T5Ranker()
    ])
    """

        Path(output_path).mkdir(parents=True, exist_ok=True)

        for conversational_turn in tqdm(data_generator.get_turn()):
            # print(conversational_turn)
            conversational_turn = pipeline(conversational_turn)
            turn_transcript = []
            for turn in conversational_turn.conversation_history:
                # print(turn)
                turn_transcript.append(
                    {
                        "participant": turn["participant"],
                        "utterance": turn["utterance"],
                        "type": turn["utterance_type"],
                        "parameters": turn.get("parameters"),
                    }
                )
            turn_transcript.extend(
                [
                    {
                        "participant": "System",
                        "utterance": conversational_turn.system_response,
                        "type": conversational_turn.system_response_type,
                    },
                    {
                        "participant": "User",
                        "utterance": conversational_turn.user_utterance,
                        "type": conversational_turn.user_utterance_type,
                        "parameters": conversational_turn.personalizedpara,
                    },
                ]
            )

            with open(f"{output_path}/{conversational_turn.turn_id}.json", "w") as f:  # noqa
                json.dump(turn_transcript, f, indent=4, ensure_ascii=False)

            with open(f"{base_path}/{run_name}.run", "a") as f:
                for index, document in enumerate(conversational_turn.ranking):
                    f.write(
                        f"{conversational_turn.turn_id}\tQ0\t{document.doc_id}\t{index+1}\t{document.score}\t{run_name}\n"  # noqa
                    )
            # with open(f"data/f_moreround_fewshot_combinedex/para_generated_conversations/semantic_cq_{conversational_turn.feedback_rounds}.run", "a") as f:  # noqa
            # for index, document in enumerate(conversational_turn.ranking):
            # f.write(f"{conversational_turn.turn_id}\tQ0\t{document.doc_id}\t{index+1}\t{document.score}\tsemantic_cq\n")  # noqa
