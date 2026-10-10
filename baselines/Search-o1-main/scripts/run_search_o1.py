# run_search_o1.py
import os
import sys
import json
import time
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import argparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.models.llm_model_service import ChatModel
from src.Agent.tools.semantic_tool import SemanticSearchTool

# Import prompt functions from the same directory
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))
from prompts import (
    get_multiqa_search_o1_instruction,
    get_task_instruction_openqa,
    get_webpage_to_reasonchain_instruction
)
from evaluate import extract_answer

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search O1 using vector database retrieval.")

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    return parser.parse_args()


def extract_between(text: str, start_tag: str, end_tag: str) -> str:
    """Extract text between two tags."""
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def main():
    # Load configuration from settings.yaml
    SETTINGS_FILE = os.path.join(project_root, "settings.yaml")
    config = load_config(SETTINGS_FILE)
    
    # Parse command line arguments
    args = parse_args()
    subset_num = args.subset_num
    
    # Load parameters from settings.yaml
    top_k = config['agent']['retriever']['top_k']
    input_file = config['agent']['files']['input_file']
    output_file = config['agent']['files']['output_file']
    error_file = config['agent']['files'].get('error_file', 'output/search_o1/errors.jsonl')
    
    # Initialize ChatModel using deployed vLLM service
    chat_model = ChatModel(config['models']['chat_model'])
    model_name = config['models']['chat_model']['model_name']
    
    # Initialize vector database retriever
    retriever = SemanticSearchTool(config_path=SETTINGS_FILE)
    
    # Set default parameters
    MAX_SEARCH_LIMIT = 10
    MAX_TURN = 15
    
    print('-----------------------')
    print(f'Input file: {input_file}')
    print(f'Output file: {output_file}')
    print(f'Error file: {error_file}')
    print(f'Top K: {top_k}')
    print(f'Model: {model_name}')
    print('-----------------------')
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        # Support both JSON and JSONL formats
        if input_file.endswith('.jsonl'):
            filtered_data = [json.loads(line) for line in f]
        else:
            filtered_data = json.load(f)
    
    if subset_num != -1:
        filtered_data = filtered_data[:subset_num]
    
    print(f'Loaded {len(filtered_data)} examples.')
    
    # Create output directories
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(error_file), exist_ok=True)
    
    # Process each question with streaming write
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with open(error_file, 'w', encoding='utf-8') as err_f:
            for idx, item in enumerate(tqdm(filtered_data, desc="Processing questions")):
                try:
                    question = item.get('question', item.get('Question', ''))
                    
                    # Initialize conversation state
                    conversation_history = []
                    search_count = 0
                    executed_queries = []  # 改为列表以保持查询顺序
                    all_retrieved_docs = []  # 收集所有检索到的文档
                    
                    # Use standard multi-QA prompt instruction
                    instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
                    user_prompt = get_task_instruction_openqa(question)
                    
                    # Initialize full prompt with instruction and question
                    full_prompt = instruction + user_prompt
                    output_text = ""  # Track complete output
                    
                    # Main reasoning loop
                    for turn in range(MAX_TURN):
                        # Get LLM response using the accumulated prompt
                        response = chat_model.get_chat(
                            system_prompt="",  # Empty system prompt
                            user_prompt=full_prompt
                        )
                        
                        if response is None:
                            print(f"Warning: LLM timeout for question {idx}")
                            break
                        
                        # Append response to prompt and output
                        full_prompt += response
                        output_text += response
                        
                        # Check if there's a search query
                        search_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                        
                        if search_query and search_count < MAX_SEARCH_LIMIT and search_query not in executed_queries:
                            # Execute search using vector database
                            search_results = retriever.execute(query=search_query, top_k=top_k)
                            
                            # 收集检索到的文档
                            all_retrieved_docs.extend(search_results)
                            
                            # Format search results
                            formatted_results = f"\n{BEGIN_SEARCH_RESULT}\n"
                            formatted_results += f"Search results for: {search_query}\n\n"
                            
                            for i, result in enumerate(search_results, 1):
                                formatted_results += f"Result {i}:\n"
                                formatted_results += f"Title: {result.get('title', 'N/A')}\n"
                                formatted_results += f"Text: {result.get('text', 'N/A')}\n"
                                formatted_results += f"Distance: {result.get('distance', 'N/A'):.4f}\n\n"
                            
                            formatted_results += f"{END_SEARCH_RESULT}\n"
                            
                            # Update state
                            search_count += 1
                            executed_queries.append(search_query)  # 添加到列表
                            
                            # Append search results to prompt
                            full_prompt += formatted_results
                            output_text += formatted_results
                            
                        elif search_query and search_count >= MAX_SEARCH_LIMIT:
                            # Search limit reached
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            full_prompt += limit_message
                            output_text += limit_message
                            
                        elif search_query and search_query in executed_queries:
                            # Duplicate query
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                            full_prompt += limit_message
                            output_text += limit_message
                            
                        else:
                            # No search query found - assume final answer
                            break
                    
                    # 收集 supporting_facts（检索到的文档标题列表）
                    supporting_facts = []
                    for doc in all_retrieved_docs:
                        title = doc.get('title', '')
                        if title and title not in supporting_facts:  # 去重
                            supporting_facts.append(title)
                    
                    # 从完整推理过程中提取最终答案
                    final_answer = extract_answer(output_text, mode='qa')
                    
                    # Store result - 使用与 NavieRAG.py 一致的格式
                    result_item = {
                        'question': question,
                        'result': final_answer,  # 只保存最终答案
                        'supporting_facts': supporting_facts,
                        'sub_question': executed_queries  # 保存生成的查询问题列表
                    }
                    
                    # 立即写入结果文件
                    out_f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                    out_f.flush()  # 立即刷新到磁盘
                    success_count += 1
                    
                except Exception as e:
                    print(f"\n✗ Error processing question {idx}: {e}")
                    import traceback
                    
                    # 获取完整的错误堆栈信息
                    error_traceback = traceback.format_exc()
                    traceback.print_exc()
                    
                    # 构建错误记录
                    error_record = {
                        'index': idx,
                        'original_data': item,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'traceback': error_traceback
                    }
                    
                    # 写入错误文件
                    err_f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                    err_f.flush()
                    error_count += 1
                    
                    print(f"✗ Error logged to {error_file}")
                    continue
    
    total_time = time.time() - start_time
    print(f'\n-----------------------')
    print(f'Processing complete!')
    print(f'Total processed: {len(filtered_data)}')
    print(f'Success: {success_count}')
    print(f'Errors: {error_count}')
    print(f'Total time: {total_time:.2f}s')
    print(f'Average time per question: {total_time/len(filtered_data):.2f}s')
    print(f'Results saved to: {output_file}')
    if error_count > 0:
        print(f'Errors saved to: {error_file}')
    print('-----------------------')


if __name__ == "__main__":
    main()
