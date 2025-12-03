from langgraph.graph import StateGraph, END
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import TypedDict, List
from pydantic import BaseModel
# import torch

from ingestion import PDFIngestor

class GraphState(TypedDict):
    question: str
    response: str
    documents: List[str]

class PdfChat:
    def __init__(self, retriever):
        model_id = "google/flan-t5-small"
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     device_map="cpu",
        #     # Add torch_dtype to specify the memory format for stability
        #     torch_dtype=torch.float32,
        #     local_files_only=False
        # )

        pipe = pipeline(
            "text2text-generation", # NOTE: Must change to text2text-generation for T5 models
            model=model_id,
            device=-1,
            max_new_tokens=250
        )
        self.model = HuggingFacePipeline(pipeline=pipe)

        builder = StateGraph(GraphState)
        builder.add_node("retrieve", self.retrieve_node)
        #builder.add_node("boost_question", self.boost_question)
        #builder.add_node("structer_document", self.structer_document)
        builder.add_node("generate_with_rag", self.generate_with_doc)
        #builder.add_node("generate", self.generate_wo_doc)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate_with_rag")
        builder.add_edge("generate_with_rag", END)
        # builder.add_conditional_edges(
        #     "boost_question",
        #     self.decide_retrieve,
        #     {
        #         "retrieve": "retrieve",
        #         "generate": "generate"
        #     }
        # )
        # builder.add_edge("retrieve", "structer_document")
        # builder.add_edge("structer_document", "generate_with_rag")
        # builder.add_edge("generate_with_rag", END)
        # builder.add_edge("generate", END)

        self.retriever = retriever

        self.graph = builder.compile()

        #self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        self.memory = ConversationBufferMemory()

    # def decide_retrieve(self, state: GraphState):
    #     question = state["question"]
    #     memory = self.memory.load_memory_variables({})
    #     source: RouteQuery = route(self.model, memory).invoke({"question": question})
    #     if source.datasource == "vectorstore":
    #         return "retrieve"
    #     else:
    #         return "generate"

    # def boost_question(self, state: GraphState):
    #     question = state["question"]
    #     memory = self.memory.load_memory_variables({})
    #     prompt = """You are an assistant in a question-answering tasks.
    #                 You have to boost the question to help search in vectorstore.
    #                 Don't make up random names.
    #                 Return a better structred question for vectorstore search, but don't make it longer
    #                 \n
    #                 Conversation history: {memory}
    #                 \n
    #                 Question: {question}
    #             """
    #     prompt = PromptTemplate.from_template(prompt)
    #     chain = prompt | self.model | StrOutputParser()

    #     question = chain.invoke({"question": question, "memory": memory})

    #     return {"question": question}


    def retrieve_node(self, state: GraphState):
        question = state["question"]
        documents = self.retriever.invoke(question)
        if not documents:
            return {"response": "I couldn't find any relevant documents. Can you please rephrase your question?"}
        return {"documents": documents}

    # def structer_document(self, state: GraphState):
    #     documents = state["documents"]
    #     question = state["question"]
    #     documents = [doc.page_content for doc in documents]

    #     prompt = """You are an expert assistant for question-answering tasks.
    #                 You have to restructure the documents for the question.
    #                 Keep it short, only knowledge that is relevant to the question.
    #                 Don't make up random names.
    #                 Return a better structured document for better understanding.
    #                 \n
    #                 Documents: {documents}
    #                 \n
    #                 Question: {question}
    #             """
    #     prompt = PromptTemplate.from_template(prompt)
    #     chain = prompt | self.model | StrOutputParser()

    #     document = chain.invoke({"question": question, "documents": documents})

    #     return  {"documents": document}

    def generate_with_doc(self, state: GraphState):
        documents = state["documents"]
        question = state["question"]
        memory = self.memory.load_memory_variables({})

        prompt = f""""You are an expert assistant for question-answering tasks.
                    Use the provided documents as context to extract and answer the question.
                    Don't be lazy, check every details in the Context.
                    If the answer is not mentioned in context, respond with 'I don't know.'
                    Keep your limited to three sentences.
                    \n
                    Conversation history: {memory}
                    \n
                    Context: {documents}
                    \n
                    Question: {question}
                    \n
                    Answer:
                """

        response = self.model.invoke(prompt)

        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
            # Fallback for the complex structure (if the pipeline returns it)
            generated_text = response[0]['generated_text']
        else:
            # Most likely scenario: the output is a single string
            generated_text = str(response)

        clean_response = generated_text.replace(prompt, '').strip()
        self.memory.save_context(inputs={"input": question}, outputs={"output": clean_response})

        return {"response": clean_response}

        # prompt = PromptTemplate.from_template(prompt)
        # chain = prompt | self.model | StrOutputParser()

        # response = chain.invoke({"memory": memory, "question": question, "context": documents})

        # self.memory.save_context(inputs={"input": question}, outputs={"output": response})

        # return {"response": response}

    # def generate_wo_doc(self, state: GraphState):
    #     question = state["question"]
    #     prompt = """You are an assistant for question-answering tasks.
    #                 If you don't know the answer, just say that you don't know.
    #                 Don't forget to check previous conversations for context.
    #                 Use three sentences maximum and keep the answer concise.
    #                 Conversation history: {memory}
    #                 Question: {question}
    #             """
    #     memory = self.memory.load_memory_variables({})
    #     prompt = PromptTemplate.from_template(prompt)
    #     chain = prompt | self.model | StrOutputParser()

    #     response = chain.invoke({"memory": memory, "question": question})
    #     self.memory.save_context(inputs={"input": question}, outputs={"output": response})

    #     return {"response": response}
