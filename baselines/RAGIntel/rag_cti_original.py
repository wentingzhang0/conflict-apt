# Code for RAGIntell System.. by Abeer Alhuthali (Alhuzali)-*-

import os
import pandas as pd
import time
from pathlib import Path
import torch
from tqdm import tqdm
from typing import List, Dict, Any

# Updated LangChain imports
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
# Core LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever

from dataclasses import dataclass, field

from flashrank import Ranker, RerankRequest
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

# Transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Ragas
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ContextPrecision, AnswerRelevancy, NoiseSensitivity
from ragas import EvaluationDataset



import os


def load_html_files(directory):
    """Load HTML files from directory and all subdirectories"""
    documents = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".html"):
                file_path = os.path.join(root, filename)
                try:
                    loader = UnstructuredHTMLLoader(file_path)
                    loaded_docs = loader.load()
                    if loaded_docs:
                        documents.extend(loaded_docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return documents

def split_documents(documents):
    """Parent-child document splitting"""
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    parent_docs = parent_splitter.split_documents(documents)
    child_docs = []

    for parent in parent_docs:
        _child_docs = child_splitter.split_documents([parent])
        for child in _child_docs:
            child.metadata["parent_id"] = str(hash(parent.page_content))
        child_docs.extend(_child_docs)

    return parent_docs, child_docs




# --- wrapper for Nomic model ---
class NomicEmbeddingWrapper(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"search_document: {text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"search_query: {text}")






# --- Modern hybrid retriever with reranking ---
def create_hybrid_retriever(texts):
    """Modern Hybrid Retriever: Dense (Nomic) + BM25 + Reranking"""

    # 1. Nomic embeddings with prefixes
    embeddings = NomicEmbeddingWrapper(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True}
    )

    # 2. Dense retriever
    vector_store = FAISS.from_documents(texts, embeddings)
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # 3. BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 10

    # 4. Hybrid retriever function
    def hybrid_retriever(query: str) -> List[Document]:
        dense_docs = dense_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)
        combined = dense_docs + bm25_docs
        return list({doc.page_content: doc for doc in combined}.values())

    # 5. Create the reranker
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    compressor = FlashrankRerank() #reranker=reranker



    # 6. Build the compressed retriever
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=RunnableLambda(hybrid_retriever)
    )
    return compressed_retriever



def load_llms():
    """Load all LLMs with proper configuration"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    llm_dict = {}


    # OpenAI
    llm_dict["openai-gpt-3.5-turbo"] = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)
    llm_dict["openai-gpt-4o"] = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)

    # FLAN-T5
    flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
    flan_t5_pipe = pipeline(
        "text2text-generation",
        model=flan_t5_model,
        tokenizer=flan_t5_tokenizer,
        max_new_tokens=200
    )
    llm_dict["flan-t5"] = HuggingFacePipeline(pipeline=flan_t5_pipe)

    # Facebook BART
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)
    bart_pipe = pipeline("text2text-generation", model=bart_model, tokenizer=bart_tokenizer, max_length=512, truncation=True)
    llm_dict["bart"] = HuggingFacePipeline(pipeline=bart_pipe)



    return {k: v for k, v in llm_dict.items() if v is not None}

def create_rag_system(retriever, llms):
    print("Create RAG systems with proper invocation")
    rag_systems = {}

    for llm_name, llm in llms.items():
        if llm is None:
            print(f"LLM {llm_name} is None, skipping.")
            continue

        print(f"Setting up RAG system for: {llm_name}")

        def invoke(input_dict, llm=llm, retriever=retriever):
            query = input_dict["query"]
            docs = retriever.invoke(query)#docs = retriever(query)
            context = "\n".join(doc.page_content for doc in docs)

            print(f"Invoking LLM {llm_name} with context:\n{context}")
            full_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

            if isinstance(llm, ChatOpenAI):
                raw_response = llm.invoke([HumanMessage(content=full_prompt)])
            else:
                raw_response = llm.invoke(full_prompt)

            # Extract only the actual LLM generated answer
            if isinstance(raw_response, dict) and "result" in raw_response:
                answer = raw_response["result"]
            elif hasattr(raw_response, "content"):
                answer = raw_response.content
            else:
                answer = str(raw_response)

            print(f"LLM answer: {answer}")

            return {
                "result": answer.strip(),
                "source_documents": docs
            }

        rag_systems[llm_name] = {"invoke": invoke}

    return rag_systems

def initialize_response_columns(df, llm_names):
    """Ensure proper dtype for response columns without overwriting"""
    try:
        llm_names = list(llm_names)  # Ensure it's a list
        for llm_name in llm_names:
            col_name = f"response_{llm_name}"
            if col_name not in df.columns:
                df[col_name] = pd.Series("", index=df.index).astype(str)
        return df
    except Exception as e:
        print(f"Error initializing columns: {str(e)}")
        raise

def check_missing_responses(df, llm_names):
    """Identify missing responses"""
    missing_responses = {}
    for index, row in df.iterrows():
        query = row["query"]
        if pd.isna(query) or not str(query).strip():
            continue

        missing_for_query = []
        for llm_name in llm_names:
            response_col = f"response_{llm_name}"
            if response_col not in df.columns:
                missing_for_query.append(llm_name)
            elif pd.isna(row[response_col]) or str(row[response_col]).strip() == "" or "Error" in str(row[response_col]):
                missing_for_query.append(llm_name)

        if missing_for_query:
            missing_responses[index] = missing_for_query
    return missing_responses


def process_missing_responses(df, rag_systems, missing_responses):
    """Process missing responses with progress tracking and robust error handling"""
    # Validate inputs
    if not isinstance(missing_responses, dict):
        raise ValueError("missing_responses must be a dictionary")

    if not all(isinstance(k, int) for k in missing_responses.keys()):
        raise ValueError("missing_responses keys must be integer indices")

    processed_count = 0
    error_count = 0

    for index, missing_llms in tqdm(missing_responses.items(), desc="Processing missing responses"):
        # Skip if index doesn't exist in DataFrame
        if index not in df.index:
            error_count += 1
            continue

        query = df.at[index, "query"]

        for llm_name in missing_llms:
            # Skip if LLM isn't in our RAG systems
            if llm_name not in rag_systems:
                df.at[index, f"response_{llm_name}"] = "Error: LLM not available"
                error_count += 1
                continue

            try:
                # Process the query
                response = rag_systems[llm_name]["invoke"]({"query": query})
                result = str(response.get("result", ""))
                df.at[index, f"response_{llm_name}"] = result
                processed_count += 1

            except Exception as e:
                error_msg = f"Error processing {llm_name}: {str(e)}"
                df.at[index, f"response_{llm_name}"] = error_msg
                error_count += 1

    # Print summary statistics
    print(f"\nProcessing complete:")
    print(f"- Total missing responses processed: {processed_count}")
    print(f"- Errors encountered: {error_count}")

    return df

import re


def clean_mitre_ids(text: str) -> str:
    """
    Extracts MITRE Technique IDs from text and returns them as a comma-separated string.
    Always returns a string, never NaN or list.
    """
    if not isinstance(text, str):
        return "N/A"
    ids = re.findall(r'T\d{4}(?:\.\d{3})?', text)
    return ", ".join(ids) if ids else "N/A"

def preprocess_ragas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans all 'response' columns in the DataFrame to contain only MITRE IDs.
    """
    df = df.copy()
    response_columns = [col for col in df.columns if "response" in col.lower()]

    if not response_columns:
        print(" No 'response' columns found in the DataFrame.")
        return df

    for col in response_columns:
        print(f"Cleaning MITRE IDs in column: {col}")
        df[col] = df[col].apply(clean_mitre_ids)

    return df


def evaluate_with_ragas(df, retriever):
    """Evaluate with Ragas metrics"""
    evaluation_data = []


    for index, row in df.iterrows():
        try:
            query = str(row["query"])
            ground_truth = str(row["ground_truth"]) if pd.notna(row["ground_truth"]) else ""

            contexts = [str(doc.page_content) for doc in retriever.invoke(query)]

            for col in df.columns:
                if col.startswith("response_"):
                    llm_name = col.replace("response_", "")
                    response = str(row[col]) if pd.notna(row[col]) else ""

                    evaluation_data.append({
                        "user_input": query,
                        "reference": ground_truth,
                        "retrieved_contexts": contexts,
                        "response": response,
                        "llm_name": llm_name
                    })
        except Exception as e:
            print(f"Skipping row {index}: {str(e)}")



    if not evaluation_data:
        raise ValueError("No valid responses for evaluation")

    ragas_dataset = EvaluationDataset.from_list(evaluation_data)
    #print("RAGAS d data:", evaluation_data)
    metrics = [
        LLMContextRecall(),
        ContextPrecision(),
        Faithfulness(),
        FactualCorrectness(),
        AnswerRelevancy(),
        NoiseSensitivity(),
    ]



    try:
        results = evaluate(ragas_dataset, metrics=metrics)

        results_df = pd.DataFrame({
            "user_input": [d["user_input"] for d in evaluation_data],
            "reference": [d["reference"] for d in evaluation_data],
            "retrieved_contexts": [d["retrieved_contexts"] for d in evaluation_data],
            "response": [d["response"] for d in evaluation_data],
            "llm_name": [d["llm_name"] for d in evaluation_data],
            **{metric.name: results[metric.name] for metric in metrics }

        })


    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        results_df = pd.DataFrame(evaluation_data)

    return results_df

def save_evaluation_results(results_df, output_file):
    """Save evaluation results"""
    results_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")



def main():
    # === Configuration ===
    excel_file = "..path to dataset in xlsx.."  #<--- update this
    output_file = "RAG-evaluation-results.xlsx" 
    html_directory = "..path to Knowlage base HTML files.." #<--- update this
    timing_output_file = "timing_summary.txt"

    timings = {}

    # === Load and process data ===
    print("Loading data...")
    start = time.time()
    df = pd.read_excel(excel_file).dropna(subset=["query"])
    df = df[df["query"].str.strip() != ""]
    timings["Data Loading"] = time.time() - start

    # === Load HTML files ===
    start = time.time()
    documents = load_html_files(html_directory)
    _, child_docs = split_documents(documents)
    timings["HTML Loading and Splitting"] = time.time() - start

    # === Create retriever ===
    print("Creating retriever...")
    start = time.time()
    retriever = create_hybrid_retriever(child_docs)
    timings["Retriever Creation"] = time.time() - start

    # === Load LLMs ===
    print("Loading LLMs...")
    start = time.time()
    llms = load_llms()
    if llms:
        print("Loaded LLMs:")
        for name, model in llms.items():
            print(f"- {name}: {model}")
    else:
        print("No LLMs were loaded.")
    timings["LLM Loading"] = time.time() - start

    # === Initialize response columns ===
    start = time.time()
    df = initialize_response_columns(df, llms.keys())
    timings["Initialize Response Columns"] = time.time() - start

    # === Build RAG systems ===
    print("Building RAG systems...")
    start = time.time()
    rag_systems = create_rag_system(retriever, llms)
    timings["RAG System Building"] = time.time() - start

    # === Process Queries ===
    start = time.time()
    missing_responses = check_missing_responses(df, llms.keys())
    if missing_responses:
        print(f"Processing {len(missing_responses)} queries...")
        df = process_missing_responses(df, rag_systems, missing_responses)
        df.to_excel(excel_file, index=False)
    timings["Query Processing"] = time.time() - start

    # === Evaluation ===
    print("Preprocessing for evaluation...")
    df = preprocess_ragas_data(df)
    start = time.time()
    print("Evaluating with RAGAS...")
    results_df = evaluate_with_ragas(df, retriever)
    save_evaluation_results(results_df, output_file)
    timings["Evaluation"] = time.time() - start


    # === Total Time ===
    total_time = sum(timings.values())
    timings["Total Runtime"] = total_time

    # === Save timings to file ===
    with open(timing_output_file, "w") as f:
        for key, value in timings.items():
            f.write(f"{key}: {value:.2f} seconds\n")

    print(f"Timing summary saved to: {timing_output_file}")

if __name__ == "__main__":
    main()