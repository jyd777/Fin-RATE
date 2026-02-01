#!/usr/bin/env python3
import json
import os
import glob
from openai import AzureOpenAI, OpenAI
from ddgs import DDGS
import argparse
from typing import List, Dict, Any
import time
import re

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None


class _DummyPbar:
    def __init__(self, total: int | None = None, desc: str | None = None, unit: str | None = None, **_: Any):
        self.total = total
        self.desc = desc
        self.unit = unit

    def update(self, n: int = 1) -> None:
        return

    def close(self) -> None:
        return

    def set_postfix(self, *_: Any, **__: Any) -> None:
        return

    def set_description(self, *_: Any, **__: Any) -> None:
        return


def _pbar(*, total: int | None, desc: str, unit: str = "it", **kwargs: Any):
    if tqdm is None:
        return _DummyPbar(total=total, desc=desc, unit=unit, **kwargs)
    return tqdm(total=total, desc=desc, unit=unit, **kwargs)


def _log(msg: str) -> None:
    if tqdm is not None:
        try:
            tqdm.write(msg)
            return
        except Exception:
            pass
    print(msg)

def _try_get_tokenizer():
    """
    Best-effort tokenizer for approximate prompt sizing.
    Uses tiktoken if available; otherwise returns None.
    """
    try:
        import tiktoken  # type: ignore

        # cl100k_base is a decent default for GPT-4/5 family token counting.
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _estimate_tokens(text: str) -> int:
    """
    Estimate tokens in text. Prefers tiktoken; falls back to a conservative heuristic.
    """
    enc = _try_get_tokenizer()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Conservative heuristic: ~4 chars/token for English-ish text; SEC filings can be dense,
    # so we keep the estimate simple and safe.
    return max(1, (len(text) + 3) // 4)


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n\n...[TRUNCATED]...\n\n"
    # Keep both ends to preserve potentially relevant headers + conclusions.
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[:max_chars]
    head = keep * 7 // 10
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to at most max_tokens (best effort).
    """
    if max_tokens <= 0:
        return ""
    enc = _try_get_tokenizer()
    if enc is not None:
        try:
            toks = enc.encode(text)
            if len(toks) <= max_tokens:
                return text
            return enc.decode(toks[:max_tokens])
        except Exception:
            pass
    # Fallback: approximate token->char conversion
    return _truncate_middle(text, max_chars=max_tokens * 4)


def _apply_max_input_tokens(prompt: str, *, max_input_tokens: int) -> str:
    """
    Enforce a maximum input token budget by truncating the Context section first.
    If Context isn't present, truncates the whole prompt.
    """
    if max_input_tokens is None or max_input_tokens <= 0:
        return prompt

    est = _estimate_tokens(prompt)
    if est <= max_input_tokens:
        return prompt

    ctx_tag = "Context:\n"
    idx = prompt.find(ctx_tag)
    if idx == -1:
        return _truncate_text_to_tokens(prompt, max_input_tokens)

    prefix = prompt[: idx + len(ctx_tag)]
    context = prompt[idx + len(ctx_tag) :]

    prefix_tokens = _estimate_tokens(prefix)
    remaining = max_input_tokens - prefix_tokens
    if remaining <= 0:
        return _truncate_text_to_tokens(prompt, max_input_tokens)

    new_context = _truncate_text_to_tokens(context, remaining)
    new_prompt = prefix + new_context
    # If still oversized due to estimation mismatch, truncate the whole thing as a final safety net.
    if _estimate_tokens(new_prompt) > max_input_tokens:
        new_prompt = _truncate_text_to_tokens(new_prompt, max_input_tokens)
    return new_prompt

def check_gpu_info():
    """check gpu info"""
    print("=" * 50)
    print("check gpu info")
    print("=" * 50)
    
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Current CUDA device name: {torch.cuda.get_device_name()}")
            print(f"CUDA_VISIBLE_DEVICES environment variable: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
            
            # Show usable GPUs
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA not available")
    except ImportError:
        print("PyTorch not installed, cannot get CUDA info")
    
    print("=" * 50)

def query_gpt4(
    prompt: str,
    deployment_name: str = "gpt-4.1",
    reasoning_effort: str | None = "medium",
    max_input_tokens: int | None = None,
) -> str:
    """
    call GPT model via OpenAI API
    
    Args:
        prompt: input prompt
        deployment_name: deployment name (e.g., "gpt-4.1", "gpt-5", "gpt-5.1")
        reasoning_effort: reasoning effort for reasoning-capable models (e.g. "low"|"medium"|"high");
          set to None to omit reasoning controls
        
    Returns:
        generated response text
    """
    try:
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

        is_gpt5_family = deployment_name.startswith("gpt-5")

        if is_gpt5_family:
            endpoint = "https://chronosense.openai.azure.com/openai/v1"
            client = OpenAI(
                base_url=endpoint,
                api_key=api_key
            )
        else:
            endpoint = os.getenv("ENDPOINT_URL", "https://chronosense.openai.azure.com/")
            api_version = "2025-01-01-preview"

            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                max_retries=5,
            )
        
        if max_input_tokens is not None:
            prompt = _apply_max_input_tokens(prompt, max_input_tokens=max_input_tokens)

        messages = [{"role": "user", "content": prompt}]
        
        completion_params = {
            "model": deployment_name,
            "messages": messages,
            "stop": None,
            "stream": False,
        }

        if is_gpt5_family:
            completion_params["max_completion_tokens"] = 8192
        else:
            completion_params["max_tokens"] = 8192
            completion_params["temperature"] = 0.7
            completion_params["top_p"] = 0.95
            completion_params["frequency_penalty"] = 0
            completion_params["presence_penalty"] = 0

        # Prefer Responses API for reasoning-capable models when available, because it supports
        # explicit reasoning controls. Fall back to Chat Completions if not supported by the endpoint.
        if is_gpt5_family:
            try:
                responses_params: Dict[str, Any] = {
                    "model": deployment_name,
                    "input": prompt,
                    "max_output_tokens": completion_params["max_completion_tokens"],
                }
                if reasoning_effort is not None:
                    responses_params["reasoning"] = {"effort": reasoning_effort}

                resp = client.responses.create(**responses_params)
                text = resp.output_text
            except Exception:
                completion = client.chat.completions.create(**completion_params)
                text = completion.choices[0].message.content
        else:
            completion = client.chat.completions.create(**completion_params)
            text = completion.choices[0].message.content

        if text:
            text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE)
        return text
    except Exception as e:
        print(f"Error calling OpenAI API for {deployment_name}: {e}")
        return f"Error calling OpenAI API for {deployment_name}: {e}"

def perform_web_search(query: str, num_results: int = 3) -> str:
    """
    Perform a web search using DuckDuckGo and return the top results.
    
    Args:
        query: The search query.
        num_results: The number of results to return.
        
    Returns:
        A string containing the concatenated search result snippets.
    """
    print(f"Performing web search for: {query}")
    try:
        # Note: you may need to install duckduckgo-search
        # pip install -U duckduckgo-search
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            if not results:
                print("No web search results found.")
                return "No results found."
            
            snippets = [f"Title: {res['title']}\nSnippet: {res['body']}" for res in results]
            return '\n\n'.join(snippets)
    except Exception as e:
        print(f"Error during web search: {e}")
        return f"Error during web search: {e}"

class CorpusLoader:
    def __init__(self, corpus_path: str = "/home/junrong/evaluation/qa/enhanced_corpus_new.jsonl"):
        """
        initialize corpus loader
        
        Args:
            corpus_path: path to corpus jsonl file
        """
        self.corpus_path = corpus_path
        self.corpus_data = {}
        self._load_corpus()
    
    def _load_corpus(self):
        """load corpus data into memory"""
        print(f"loading corpus from: {self.corpus_path}")
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if '_id' in data:
                            self.corpus_data[data['_id']] = data.get('text', '')
                    except json.JSONDecodeError as e:
                        print(f"warning: skip invalid json at line {line_num}: {e}")
                        continue
            print(f"loaded {len(self.corpus_data)} documents from corpus")
        except FileNotFoundError:
            print(f"warning: corpus file not found: {self.corpus_path}")
            self.corpus_data = {}
        except Exception as e:
            print(f"error loading corpus: {e}")
            self.corpus_data = {}
    
    def get_text_by_ids(self, doc_ids: List[str]) -> str:
        """
        get text content by document ids
        
        Args:
            doc_ids: list of document ids
            
        Returns:
            concatenated text content
        """
        texts = []
        for doc_id in doc_ids:
            if doc_id in self.corpus_data:
                texts.append(self.corpus_data[doc_id])
            else:
                print(f"warning: document id {doc_id} not found in corpus")
        
        return '\n\n'.join(texts)

class GPT4AnswerGenerator:
    def __init__(
        self,
        deployment_name: str = "gpt-4.1",
        corpus_loader: CorpusLoader = None,
        web_search: bool = False,
        max_input_tokens: int = 260_000,
    ):
        """
        initialize GPT-4 Turbo model via Azure
        
        Args:
            deployment_name: Azure deployment name
            corpus_loader: corpus loader instance
            web_search: enable web search mode
        """
        self.deployment_name = deployment_name
        self.corpus_loader = corpus_loader
        self.web_search = web_search
        self.max_input_tokens = max_input_tokens
        print(f"initializing GPT-4 Turbo model via Azure deployment: {deployment_name}")
    
    def _get_content_for_qa(self, qa_pair: Dict[str, Any]) -> str:
        """
        get content for QA pair, either from content field or from corpus using doc_id/doc_ids
        
        Args:
            qa_pair: QA pair dictionary
            
        Returns:
            content string
        """
        # if content field exists, use it directly
        if 'content' in qa_pair and qa_pair['content']:
            return qa_pair['content']
        
        # if no content field, try to get from corpus using doc_id/doc_ids
        doc_ids = []
        
        # check for doc_id field (single document)
        if 'doc_id' in qa_pair and qa_pair['doc_id']:
            doc_ids.append(qa_pair['doc_id'])
        
        # check for doc_ids field (multiple documents)
        if 'doc_ids' in qa_pair and qa_pair['doc_ids']:
            if isinstance(qa_pair['doc_ids'], list):
                doc_ids.extend(qa_pair['doc_ids'])
            else:
                doc_ids.append(qa_pair['doc_ids'])
        
        if doc_ids and self.corpus_loader:
            content = self.corpus_loader.get_text_by_ids(doc_ids)
            if content:
                return content
            else:
                print(f"warning: no content found for doc_ids: {doc_ids}")
                return ""
        else:
            print(f"warning: no content field and no valid doc_id/doc_ids found in QA pair")
            return ""
    
    def generate_answer(self, question: str, qa_pair: Dict[str, Any]) -> str:
        """
        generate answer based on question and QA pair
        
        Args:
            question: question
            qa_pair: QA pair dictionary
            
        Returns:
            generated answer
        """
        # get content from QA pair
        content = self._get_content_for_qa(qa_pair)
        
        prompt_context = content

        if self.web_search:
            _log(f"web search mode enabled, searching for question: {question}")
            web_results = perform_web_search(question)
            if web_results and "Error during web search" not in web_results and "No results found" not in web_results:
                if prompt_context:
                    prompt_context = f"Web Search Results:\n{web_results}\n\nCorpus Content:\n{prompt_context}"
                else:
                    prompt_context = f"Web Search Results:\n{web_results}"

        if not prompt_context:
            return "error: no content available for this question"
        
        # build prompt (force English and final answer only)
        prompt = (
            "You are a SEC filing financial analysis expert.\n"
            "- Answer in English only.\n"
            "- Output only the final answer. Do not include chain-of-thought or <think> sections.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{prompt_context}"
        )
        
        try:
            # call gpt4 api
            generated_text = query_gpt4(prompt, self.deployment_name, max_input_tokens=self.max_input_tokens)
            return generated_text
            
        except Exception as e:
            print(f"error generating answer: {e}")
            return f"error generating answer: {e}"
    
    def generate_answers_batch(self, qa_batch: List[Dict[str, Any]]) -> List[str]:
        """
        batch generate answers, improve efficiency
        
        Args:
            qa_batch: QA pairs batch
            
        Returns:
            generated answers list
        """
        answers = []
        for i, qa_pair in enumerate(qa_batch):
            try:
                question = qa_pair['question']
                answer = self.generate_answer(question, qa_pair)
                answers.append(answer)
            except Exception as e:
                print(f"error generating batch {i+1} answer: {e}")
                answers.append(f"error generating answer: {e}")
        
        return answers
    
    def process_qa_pairs(self, qa_pairs: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        process QA pairs list, generate generated_answer for each QA pair
        
        Args:
            qa_pairs: QA pairs list
            batch_size: batch size
            
        Returns:
            processed QA pairs list
        """
        processed_pairs = []
        total_pairs = len(qa_pairs)
        
        # filter out valid QA pairs
        valid_qa_pairs = []
        for qa_pair in qa_pairs:
            if 'question' in qa_pair:
                # check if has content or doc_id/doc_ids
                has_content = 'content' in qa_pair and qa_pair['content']
                has_doc_id = 'doc_id' in qa_pair and qa_pair['doc_id']
                has_doc_ids = 'doc_ids' in qa_pair and qa_pair['doc_ids']
                
                if has_content or has_doc_id or has_doc_ids:
                    valid_qa_pairs.append(qa_pair)
                else:
                    print(f"skip QA pair with missing content/doc_id/doc_ids: {qa_pair}")
                    processed_pairs.append(qa_pair)
            else:
                print(f"skip QA pair with missing question field: {qa_pair}")
                processed_pairs.append(qa_pair)
        
        # batch process
        pbar = _pbar(total=len(valid_qa_pairs), desc="Generating answers", unit="qa")
        for i in range(0, len(valid_qa_pairs), batch_size):
            batch = valid_qa_pairs[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(valid_qa_pairs) + batch_size - 1) // batch_size
            
            try:
                # batch generate answers
                generated_answers = self.generate_answers_batch(batch)
                
                # create processed QA pairs
                for j, (qa_pair, generated_answer) in enumerate(zip(batch, generated_answers)):
                    new_qa_pair = {
                        'question': qa_pair['question'],
                        'answer': qa_pair.get('answer', ''),
                        'generated_answer': generated_answer
                    }
                    # pass through identifying/context fields to avoid post-merge
                    if 'qid' in qa_pair:
                        new_qa_pair['qid'] = qa_pair['qid']
                    if 'q_id' in qa_pair:
                        new_qa_pair['q_id'] = qa_pair['q_id']
                    if 'key_points' in qa_pair:
                        new_qa_pair['key_points'] = qa_pair['key_points']
                    processed_pairs.append(new_qa_pair)
                pbar.update(len(batch))
                pbar.set_postfix(batch=f"{batch_num}/{total_batches}")
                
                # add delay to avoid API rate limits (sleep after every batch except the last)
                if batch_num < total_batches:
                    _log("\nPausing for 60 seconds after this batch to avoid rate limits...\n")
                    time.sleep(120)
                
            except Exception as e:
                print(f"✗ error processing batch {batch_num}: {e}")
                for qa_pair in batch:
                    processed_pairs.append(qa_pair)
                pbar.update(len(batch))
        pbar.close()
        
        return processed_pairs

def process_json_file(file_path: str, generator: GPT4AnswerGenerator, output_dir: str, batch_size: int = 5) -> str:
    """
    process single JSON file
    
    Args:
        file_path: JSON file path
        generator: GPT4 answer generator
        output_dir: output directory path
        
    Returns:
        output file path
    """
    print(f"processing file: {file_path}")
    
    # read original file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # generate output file path
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_with_gpt4_answers.json")
    
    # stream results as they are generated
    with open(output_path, 'w', encoding='utf-8') as out_f:
        if isinstance(data, list):
            # Stream a JSON array of items; each item has {question, answer, generated_answer}
            out_f.write("[\n")
            first_written = False
            total = len(data)
            pbar = _pbar(total=total, desc=f"{base_name}", unit="qa")
            for i in range(0, total, batch_size):
                batch = data[i:i + batch_size]
                try:
                    generated_answers = generator.generate_answers_batch(batch)
                except Exception as e:
                    print(f"error generating batch {i // batch_size + 1}: {e}")
                    generated_answers = ["error generating answer"] * len(batch)
                for qa_pair, gen_ans in zip(batch, generated_answers):
                    new_qa_pair = {
                        'question': qa_pair.get('question', ''),
                        'answer': qa_pair.get('answer', ''),
                        'generated_answer': gen_ans
                    }
                    # pass through identifying/context fields to avoid post-merge
                    if isinstance(qa_pair, dict):
                        if 'qid' in qa_pair:
                            new_qa_pair['qid'] = qa_pair['qid']
                        if 'q_id' in qa_pair:
                            new_qa_pair['q_id'] = qa_pair['q_id']
                        if 'key_points' in qa_pair:
                            new_qa_pair['key_points'] = qa_pair['key_points']
                    if first_written:
                        out_f.write(",\n")
                    out_f.write(json.dumps(new_qa_pair, ensure_ascii=False, indent=2))
                    out_f.flush()
                    first_written = True
                pbar.update(len(batch))
                # sleep after every batch except the last
                if i + batch_size < total:
                    _log("\nPausing for 60 seconds after this batch to avoid rate limits...\n")
                    time.sleep(60)
            pbar.close()
            out_f.write("\n]\n")
        elif isinstance(data, dict) and 'qa_pairs' in data and isinstance(data['qa_pairs'], list):
            # Stream an object with other top-level fields + a streaming qa_pairs array
            out_f.write("{\n")
            other_keys = [k for k in data.keys() if k != 'qa_pairs']
            for idx, k in enumerate(other_keys):
                out_f.write(f"  {json.dumps(k)}: ")
                out_f.write(json.dumps(data[k], ensure_ascii=False, indent=2))
                out_f.write(",\n")
            out_f.write('  "qa_pairs": [\n')
            first_written = False
            qa_list = data['qa_pairs']
            pbar = _pbar(total=len(qa_list), desc=f"{base_name}", unit="qa")
            for i in range(0, len(qa_list), batch_size):
                batch = qa_list[i:i + batch_size]
                try:
                    generated_answers = generator.generate_answers_batch(batch)
                except Exception as e:
                    print(f"error generating batch {i // batch_size + 1}: {e}")
                    generated_answers = ["error generating answer"] * len(batch)
                for qa_pair, gen_ans in zip(batch, generated_answers):
                    new_qa_pair = {
                        'question': qa_pair.get('question', ''),
                        'answer': qa_pair.get('answer', ''),
                        'generated_answer': gen_ans
                    }
                    # pass through identifying/context fields to avoid post-merge
                    if isinstance(qa_pair, dict):
                        if 'qid' in qa_pair:
                            new_qa_pair['qid'] = qa_pair['qid']
                        if 'q_id' in qa_pair:
                            new_qa_pair['q_id'] = qa_pair['q_id']
                        if 'key_points' in qa_pair:
                            new_qa_pair['key_points'] = qa_pair['key_points']
                    item_str = json.dumps(new_qa_pair, ensure_ascii=False, indent=2)
                    item_str = "\n".join("    " + line for line in item_str.splitlines())
                    if first_written:
                        out_f.write(",\n")
                    out_f.write(item_str)
                    out_f.flush()
                    first_written = True
                pbar.update(len(batch))
                # sleep after every batch except the last
                if i + batch_size < len(qa_list):
                    _log("\nPausing for 60 seconds after this batch to avoid rate limits...\n")
                    time.sleep(60)
            pbar.close()
            out_f.write("\n  ]\n}\n")
        else:
            print(f"unrecognized data format: {file_path}")
            return None
    
    print(f"processing completed, output file: {output_path}")
    return output_path

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Process QA files with GPT-4 Turbo')
    parser.add_argument('--input_path', help='Input directory path or single JSON file path', default="/home/yidong/qa_dataset/latest/qa_pairs_with_key_points.json")
    parser.add_argument('--output-dir', '-o', default="/home/yidong/new_datatset/gpt4_answer", 
                       help='Output directory path (default: /home/yidong/new_datatset/gpt4_answer)')
    parser.add_argument('--deployment', '-d', default="gpt-4.1", 
                       help='Azure OpenAI deployment name (default: gpt-4.1)')
    parser.add_argument('--corpus', '-c', default="/home/yidong/DRAGIN/enhanced_corpus_new.jsonl",
                       help='Corpus file path (default: /home/yidong/DRAGIN/enhanced_corpus_new.jsonl)')
    parser.add_argument('--web_search', action='store_true', help='Enable web search mode to augment context.')
    parser.add_argument(
        '--max_input_tokens',
        type=int,
        default=260_000,
        help='Maximum input tokens to send to the model (default: 260000). Oversized prompts are truncated.',
    )
    
    args = parser.parse_args()
    
    # check GPU info
    check_gpu_info()
    
    # check if input path exists
    if not os.path.exists(args.input_path):
        print(f"error: input path does not exist: {args.input_path}")
        return
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"output directory: {args.output_dir}")
    
    # initialize corpus loader
    corpus_loader = CorpusLoader(args.corpus)
    
    # initialize GPT-4 answer generator
    generator = GPT4AnswerGenerator(
        args.deployment,
        corpus_loader=corpus_loader,
        web_search=args.web_search,
        max_input_tokens=args.max_input_tokens,
    )
    
    # determine if input is directory or single file
    if os.path.isdir(args.input_path):
        # process all JSON files in directory
        json_files = glob.glob(os.path.join(args.input_path, "*.json"))
        
        if not json_files:
            print(f"no JSON files found in {args.input_path} directory")
            return
        
        print(f"found {len(json_files)} JSON files:")
        for file_path in json_files:
            print(f"  - {os.path.basename(file_path)}")
        
        # process each JSON file
        processed_files = []
        for file_path in json_files:
            try:
                output_path = process_json_file(file_path, generator, args.output_dir)
                if output_path:
                    processed_files.append(output_path)
            except Exception as e:
                print(f"error processing file {file_path}: {e}")
        
        print(f"\nprocessing completed! processed {len(processed_files)} files:")
        for output_path in processed_files:
            print(f"  - {output_path}")
    
    elif os.path.isfile(args.input_path) and args.input_path.endswith('.json'):
        # process single JSON file
        print(f"processing single file: {args.input_path}")
        try:
            output_path = process_json_file(args.input_path, generator, args.output_dir)
            if output_path:
                print(f"\nprocessing completed! output file: {output_path}")
            else:
                print("processing failed")
        except Exception as e:
            print(f"error processing file {args.input_path}: {e}")
    
    else:
        print(f"error: input path must be a directory or a JSON file: {args.input_path}")
        return

if __name__ == "__main__":
    main() 
