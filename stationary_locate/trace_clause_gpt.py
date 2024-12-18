import json
import os
import re
from pprint import pprint

import jinja2
import torch
from numpy.array_api import trunc
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import clause_utils


def generate_prompt(input_text, candidate_caps, template, tokenizer):
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


def generate_filter_prompt(text, all_relevant_laws, template, tokenizer):
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


def parse_output(output, caps_chunk):
    json_match = re.search(r'\[.*\]', output, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # print(f"json_str: {json_str}")
        try:
            relevant_caps = []
            parsed_json = eval(json_str)
            parsed_json = list(parsed_json)
            for i in range(len(caps_chunk)):
                if parsed_json[i] == 'directly_applicable':
                    relevant_caps.append(f"{caps_chunk[i]}: 适用")
                elif parsed_json[i] == 'highly_relevant':
                    relevant_caps.append(f"{caps_chunk[i]}: 高度相关")
            return relevant_caps
        except:
            # print("Error parsing JSON.")
            return []
    else:
        # print("No valid JSON content found.")
        return []


def evaluate_relevance(input_text, caps_chunks, llm, template, tokenizer):
    prompts = []
    for caps_chunk in caps_chunks:
        prompt = generate_prompt(input_text, caps_chunk, template, tokenizer)
        prompts.append(prompt)

    sampling_params = SamplingParams(max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in outputs]

    all_relevant_laws = []
    for i in range(len(res_contents)):
        res_content = res_contents[i]
        caps_chunk = caps_chunks[i]
        # print(f"for caps: {caps_chunk}")
        # print(f"res_content: {res_content}")

        relevant_laws = parse_output(res_content, caps_chunk)
        all_relevant_laws.extend(relevant_laws)
    return all_relevant_laws


def filter_laws(text, all_relevant_laws, llm, filter_template, tokenizer):
    prompt = generate_filter_prompt(text, all_relevant_laws, filter_template, tokenizer)
    sampling_params = SamplingParams(max_tokens=8192)
    output = llm.generate(prompt, sampling_params=sampling_params)
    res_content = output[0].outputs[0].text
    return res_content


def parse_laws(laws, caps):
    matched_caps = re.findall(r'cap \d+', laws.lower())
    if matched_caps:
        all_cap_numbers = [cap['cap_number'] for key, cap in caps.items()]
        all_cap_numbers = set([number.lower() for number in all_cap_numbers])
        matched_caps = set(matched_caps)
        intersected_caps = matched_caps.intersection(all_cap_numbers)
        if intersected_caps:
            relevant_cap_numbers = list(intersected_caps)
            # 首母大写
            relevant_cap_numbers = [cap.upper() for cap in relevant_cap_numbers]
            return relevant_cap_numbers
    return []


def evaluate_cap_by_part_section(text, current_caps, llm, tokenizer, part_section_template):
    toc_lst = [cap.get_part_section_table_of_contents() for cap in current_caps]
    prompts = [part_section_template.render(input_text=text, toc=toc) for toc in toc_lst]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": part_section_text}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for part_section_text in prompts
    ]
    sampling_params = SamplingParams(max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in outputs]
    relevant_sections_for_caps = []
    for i in range(len(res_contents)):
        json_match = re.search(r'\[.*\]', res_contents[i], re.DOTALL)
        cap = current_caps[i]
        if json_match:
            json_str = json_match.group(0)
            try:
                relevant_sections = []
                parsed_lst = eval(json_str)
                parsed_lst = list(parsed_lst)
                relevant_sections_for_caps.append(parsed_lst)
            except:
                # print("Error parsing JSON.")
                relevant_sections_for_caps.append([])
        else:
            # print("No valid JSON content found.")
            relevant_sections_for_caps.append([])
    return relevant_sections_for_caps


def evaluate_section(text, relevant_path, llm, tokenizer, final_section_template):
    prompts = []
    for path_dic in relevant_path:
        section_toc = path_dic["section_toc"]
        prompt = final_section_template.render(input_text=text, section_toc=section_toc)
        prompts.append(prompt)
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in prompts
    ]
    sampling_params = SamplingParams(max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    res_contents = [output.outputs[0].text for output in outputs]
    pprint(res_contents)

    final_relevant_path = []
    for i in range(len(res_contents)):
        if "YES_RELEVANT" in res_contents[i]:
            final_relevant_path.append(relevant_path[i])

    return final_relevant_path


def main():
    loader = jinja2.FileSystemLoader('template')
    env = jinja2.Environment(loader=loader)
    template = env.get_template('find_relevant_laws.jinja2')
    filter_template = env.get_template('filter_laws.jinja2')
    part_section_template = env.get_template('part_section.jinja2')
    final_section_template = env.get_template('final_section.jinja2')

    hierarchy_law = clause_utils.build_hierarchy("law_data/spider")
    caps = hierarchy_law.get_caps()
    caps_list = [f"{cap['cap_number']} {cap['title']}: {cap['long_title']}" for key, cap in caps.items()]
    chunk_size = 5
    caps_chunks = [caps_list[i:i + chunk_size] for i in range(0, len(caps_list), chunk_size)]

    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    # model_name = "/home/hansirui/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75/"
    # model_name = "/home/hansirui/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        # tensor_parallel_size=4,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    while True:
        # 清空linux shell

        text = input("请输入事实：")
        if text == "exit":
            break

        os.system('cls||clear')

        # for caps_chunk in tqdm(caps_chunks, desc="Processing caps chunks", total=len(caps_chunks)):
        all_relevant_laws = evaluate_relevance(text, caps_chunks, llm, template, tokenizer)
        # print(all_relevant_laws)
        laws = filter_laws(text, all_relevant_laws, llm, filter_template, tokenizer)
        # print(laws)
        # 抽取法条，遍历所有法条，找到相关法条
        law_numbers = parse_laws(laws, caps)
        # print(law_numbers)

        # 随后寻找具体条文
        if law_numbers:
            current_caps = []
            for law_number in law_numbers:
                current_cap = hierarchy_law.caps[law_number]
                current_caps.append(current_cap)
            if not current_caps:
                print("未找到相关法条。")
                continue

            relevant_sections = evaluate_cap_by_part_section(text, current_caps, llm, tokenizer, part_section_template)
            pprint(relevant_sections)

            relevant_path = []
            for i in range(len(relevant_sections)):
                law_number = law_numbers[i]
                relevant_section_lst = relevant_sections[i]
                current_cap = hierarchy_law.caps[law_number]
                # 从current_cap遍历所有part下的section，找到相关的section
                for part_idx, part in current_cap.parts.items():
                    for section_idx, section in part.sections.items():
                        if f"Section {section.section_number}" in relevant_section_lst:
                            # print(f"{law_number} {part.title} {section.title}: {section.content}")
                            prefix = f"Cap {law_number}: {current_cap.title}\n"
                            prefix += f"Part {part_idx}: {part.title}"
                            relevant_path.append(
                                {
                                    "law_number": law_number,
                                    'cap title': current_cap.title,
                                    "part_idx": part_idx,
                                    "part_title": part.title,
                                    "section_idx": section_idx,
                                    "section_toc": section.get_section_toc(),
                                }
                            )
            final_relevant_path = evaluate_section(text, relevant_path, llm, tokenizer, final_section_template)
            # pprint(final_relevant_path)

            # 获得other relevant path
            other_relevant_path = []
            # 首先是law_numbers中没有在final_relevant_path中的
            final_law_numbers = set([path["law_number"] for path in final_relevant_path])
            for law_number in law_numbers:
                if law_number not in final_law_numbers:
                    current_cap = hierarchy_law.caps[law_number]
                    other_relevant_path.append({
                        "law_number": law_number,
                        'cap title': current_cap.title,
                    })
            # 随后是在final_relevant_path中的，但是没有找到具体的section
            for path in relevant_path:
                if path not in final_relevant_path:
                    other_relevant_path.append(path)

            print("相关条文:")
            pprint(final_relevant_path)

            print("拓展阅读条文:")
            pprint(other_relevant_path)

            # Save to file
            if not os.path.exists("./res"):
                os.makedirs("./res", exist_ok=True)
            file_id = len(os.listdir("./res")) + 1
            with open(f"./res/relevant_path_{file_id}.json", "w", encoding="utf-8") as f:
                json_content = {
                    "question:": text,
                    "relevant_path": final_relevant_path,
                    "other_relevant_path": other_relevant_path,
                    "backend_model": model_name
                }
                json.dump(json_content, f, ensure_ascii=False, indent=4)

        else:
            print("未找到相关法条。")

        # if laws:
        #     print("相关法条:")
        #     for cap in laws:
        #         print(cap)
        # else:
        #     print("未找到相关法条。")


if __name__ == "__main__":
    main()

# On the morning of 24 August 2013, the defendant arrived in Hong Kong by air and was attempting to use a special immigration channel.\nThe defendant was told he could not use this channel and became upset. He also smelt of alcohol.\nAn immigration officer directed him towards normal clearance, and the defendant's behavior attracted the attention of three police officers who were on anti-terrorist duty.\nShortly after that, the defendant allegedly grabbed one of the sub-machine guns held by one of the police officers and pulled it and the officer towards him. The defendant was subdued and apologised.\nPW1, PW2, PW3, PW4, and PW5 witnessed the incident. PW3's evidence was that he saw the defendant gra bbing and pulling his sub-machine gun.\nA CCTV camera recorded the area, capturing the incident, though it was obscured at times.\nThe defendant’s wife and 12-year-old daughter went through immigration ahead of him, and he had travelled from Incheon International Airport with them.\nThe defendant described the incident as an attempt to apologize to the police officer, citing cultural norms as his explanation for the alleged grabbing action.\nPW3, the police officer, verified the defendant's alleged alcohol intoxication.
# How can I create a will or trust?
# How can I apply for a work visa?
# What type of business structure should I choose for my startup?
