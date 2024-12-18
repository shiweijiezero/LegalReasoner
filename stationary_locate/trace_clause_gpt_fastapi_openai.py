from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import re
from typing import List, Dict, Optional
import jinja2
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import clause_utils
from pathlib import Path

app = FastAPI(title="Legal Analysis API")

# Initialize global variables
loader = jinja2.FileSystemLoader('template')
env = jinja2.Environment(loader=loader)
template = env.get_template('find_relevant_laws.jinja2')
filter_template = env.get_template('filter_laws.jinja2')
part_section_template = env.get_template('part_section.jinja2')
final_section_template = env.get_template('final_section.jinja2')

# Load hierarchy law data
hierarchy_law = clause_utils.build_hierarchy("law_data/spider")
caps = hierarchy_law.get_caps()
caps_list = [f"{cap['cap_number']} {cap['title']}: {cap['long_title']}" for key, cap in caps.items()]
CHUNK_SIZE = 5
caps_chunks = [caps_list[i:i + CHUNK_SIZE] for i in range(0, len(caps_list), CHUNK_SIZE)]

# Initialize model
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    tensor_parallel_size=torch.cuda.device_count(),
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class LegalRequest(BaseModel):
    text: str

class LegalResponse(BaseModel):
    relevant_paths: List[Dict]
    other_relevant_paths: List[Dict]
    model_name: str

def generate_prompt(input_text: str, candidate_caps: List[str], template: jinja2.Template, tokenizer) -> str:
    prompt_text = template.render(input_text=input_text, candidate_caps=candidate_caps)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": prompt_text}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

def generate_filter_prompt(text: str, all_relevant_laws: List[str], template: jinja2.Template, tokenizer) -> str:
    prompt_text = template.render(input_text=text, candidate_caps=all_relevant_laws)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": prompt_text}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt

def parse_output(output: str, caps_chunk: List[str]) -> List[str]:
    json_match = re.search(r'\[.*\]', output, re.DOTALL)
    if json_match:
        try:
            parsed_json = eval(json_match.group(0))
            relevant_caps = []
            for i in range(len(caps_chunk)):
                if parsed_json[i] == 'directly_applicable':
                    relevant_caps.append(f"{caps_chunk[i]}: 适用")
                elif parsed_json[i] == 'highly_relevant':
                    relevant_caps.append(f"{caps_chunk[i]}: 高度相关")
            return relevant_caps
        except:
            return []
    return []

def evaluate_relevance(input_text: str, caps_chunks: List[List[str]], llm, template: jinja2.Template, tokenizer) -> List[str]:
    prompts = [generate_prompt(input_text, caps_chunk, template, tokenizer) for caps_chunk in caps_chunks]
    sampling_params = SamplingParams(max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in outputs]

    all_relevant_laws = []
    for i, res_content in enumerate(res_contents):
        relevant_laws = parse_output(res_content, caps_chunks[i])
        all_relevant_laws.extend(relevant_laws)
    return all_relevant_laws

def filter_laws(text: str, all_relevant_laws: List[str], llm, filter_template: jinja2.Template, tokenizer) -> str:
    prompt = generate_filter_prompt(text, all_relevant_laws, filter_template, tokenizer)
    sampling_params = SamplingParams(max_tokens=8192)
    output = llm.generate(prompt, sampling_params=sampling_params)
    return output[0].outputs[0].text

def parse_laws(laws: str, caps: Dict) -> List[str]:
    matched_caps = re.findall(r'cap \d+', laws.lower())
    if matched_caps:
        all_cap_numbers = set(cap['cap_number'].lower() for key, cap in caps.items())
        matched_caps = set(matched_caps)
        intersected_caps = matched_caps.intersection(all_cap_numbers)
        if intersected_caps:
            return [cap.upper() for cap in intersected_caps]
    return []

def evaluate_cap_by_part_section(text: str, current_caps: List, llm, tokenizer, part_section_template: jinja2.Template) -> List[List]:
    toc_lst = [cap.get_part_section_table_of_contents() for cap in current_caps]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": part_section_template.render(input_text=text, toc=toc)}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for toc in toc_lst
    ]

    sampling_params = SamplingParams(max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in outputs]

    relevant_sections_for_caps = []
    for res_content in res_contents:
        json_match = re.search(r'\[.*\]', res_content, re.DOTALL)
        if json_match:
            try:
                parsed_lst = eval(json_match.group(0))
                relevant_sections_for_caps.append(list(parsed_lst))
            except:
                relevant_sections_for_caps.append([])
        else:
            relevant_sections_for_caps.append([])
    return relevant_sections_for_caps

def evaluate_section(text: str, relevant_path: List[Dict], llm, tokenizer, final_section_template: jinja2.Template) -> List[Dict]:
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": final_section_template.render(input_text=text, section_toc=path["section_toc"])}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for path in relevant_path
    ]

    sampling_params = SamplingParams(max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in outputs]

    return [path for i, path in enumerate(relevant_path) if "YES_RELEVANT" in res_contents[i]]

@app.post("/analyze", response_model=LegalResponse)
async def analyze_legal_text(request: LegalRequest):
    try:
        text = request.text

        # Evaluate initial relevance
        all_relevant_laws = evaluate_relevance(text, caps_chunks, llm, template, tokenizer)
        if not all_relevant_laws:
            return LegalResponse(
                relevant_paths=[],
                other_relevant_paths=[],
                model_name=MODEL_NAME
            )

        # Filter and parse laws
        laws = filter_laws(text, all_relevant_laws, llm, filter_template, tokenizer)
        law_numbers = parse_laws(laws, caps)

        if not law_numbers:
            return LegalResponse(
                relevant_paths=[],
                other_relevant_paths=[],
                model_name=MODEL_NAME
            )

        # Get current caps
        current_caps = [hierarchy_law.caps[law_number] for law_number in law_numbers if law_number in hierarchy_law.caps]
        if not current_caps:
            return LegalResponse(
                relevant_paths=[],
                other_relevant_paths=[],
                model_name=MODEL_NAME
            )

        # Evaluate sections
        relevant_sections = evaluate_cap_by_part_section(text, current_caps, llm, tokenizer, part_section_template)

        # Build relevant paths
        relevant_path = []
        for i, relevant_section_lst in enumerate(relevant_sections):
            law_number = law_numbers[i]
            current_cap = hierarchy_law.caps[law_number]

            for part_idx, part in current_cap.parts.items():
                for section_idx, section in part.sections.items():
                    if f"Section {section.section_number}" in relevant_section_lst:
                        relevant_path.append({
                            "law_number": law_number,
                            "cap_title": current_cap.title,
                            "part_idx": part_idx,
                            "part_title": part.title,
                            "section_idx": section_idx,
                            "section_toc": section.get_section_toc(),
                        })

        # Get final relevant paths
        final_relevant_path = evaluate_section(text, relevant_path, llm, tokenizer, final_section_template)

        # Get other relevant paths
        other_relevant_path = []
        final_law_numbers = set(path["law_number"] for path in final_relevant_path)

        # Add laws not in final relevant path
        for law_number in law_numbers:
            if law_number not in final_law_numbers:
                current_cap = hierarchy_law.caps[law_number]
                other_relevant_path.append({
                    "law_number": law_number,
                    "cap_title": current_cap.title,
                })

        # Add paths not in final relevant path
        other_relevant_path.extend([path for path in relevant_path if path not in final_relevant_path])

        return LegalResponse(
            relevant_paths=final_relevant_path,
            other_relevant_paths=other_relevant_path,
            model_name=MODEL_NAME
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)