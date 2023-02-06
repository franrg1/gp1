#funcion para llamar a openai desde la api y que devuelva solamente la respuesta. Para correr "python aicontent.py"
import os
import openai
import config
import pinecone
import pandas as pd
from tqdm.auto import tqdm
import json
from openai.embeddings_utils import get_embedding
import matplotlib

import json
with open('D:/Users/julian.monis/chatgptpython/ai-content-starting-template/cuervo-mapping.json', 'r') as fp:
    mappings = json.load(fp)

def load_index():
    pinecone.init(
        api_key='d9b601c8-7d02-4c58-828e-d306cc6bc45a',  # app.pinecone.io
        environment='us-west1-gcp'
    )
    index_name = 'amzn-semantic-search'
    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")

    return pinecone.Index(index_name)

index = load_index()





openai.api_key = "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"





def create_context(question, index, max_len=3750, size="babbage"):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(question, engine=f'text-search-{size}-query-001')
    res = index.query(q_embed, top_k=5, include_metadata=True)
    

    cur_len = 0
    contexts = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    return "\n\n###\n\n".join(contexts)

instructions = {
    "conservative Q&A": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    "paragraph about a question":"Write a paragraph, addressing the question, and use the text below to obtain relevant information\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nParagraph long Answer:",
    "bullet point": "Write a bullet point list of possible answers, addressing the question, and use the text below to obtain relevant information\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nBullet point Answer:",
    "summarize problems given a topic": "Write a summary of the problems addressed by the questions below\"\n\n{0}\n\n---\n\n",
    "extract key libraries and tools": "Write a list of libraries and tools present in the context below\"\n\nContext:\n{0}\n\n---\n\n",
    "just instruction": "{1} given the common questions and answers below \n\n{0}\n\n---\n\n",
    "summarize": "Write an elaborate, paragraph long summary about \"{1}\" given the questions and answers from a public forum on this topic\n\n{0}\n\n---\n\nSummary:",
    "chat": "The following is a chat conversation between a AI ecommerce assistant and a user. Write a paragraph, addressing the question, and use the text below to obtain relevant information. If question absolutely cannot be answered based on the context, say I dont know\"\n\nContext:\n{0}\n\n---\n\nChat: {1}",
    "short answer": "Write a short answer to the question based on the context\"\n\nContext:\n{0}\n\n---\n\nChat: {1}"
}
def answer_question(
    index,
    fine_tuned_qa_model="text-davinci-003",
    question="i need a sun screen product",
    instruction="Answer the query based on the context below, and if the query can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    max_len=3550,
    size="babbage",
    debug=False,
    max_tokens=400,
    stop_sequence=None,
    # domains=["huggingface", "tensorflow", "streamlit", "pytorch"],
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        index,
        max_len=max_len,
        size=size,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # fine-tuned models requires model parameter, whereas other models require engine parameter
        model_param = (
            {"model": fine_tuned_qa_model}
            if ":" in fine_tuned_qa_model
            and fine_tuned_qa_model.split(":")[1].startswith("ft")
            else {"engine": fine_tuned_qa_model}
        )
        response = openai.Completion.create(
            prompt=instruction.format(context, question),
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            **model_param,
        )
        return response["choices"][0]["text"].strip()

    except Exception as e:
        print(e)
        return ""





