import re
import json

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # 每行都是一个独立的JSON对象，但是键值对中的引号是单引号，只替换第一个和最后一个
            # 先替换所有的双引号为单引号
            line = line[10:-3]
            data.append(line)
    return data

def parse_law_file(file_path):
    laws = []
    current_law = None

    # 读取JSONL文件
    lines = read_jsonl_file(file_path)

    for line in lines:
        text = line

        # Extract Cap. number with optional sub. leg.
        print(text)
        cap_match = re.search(r'Cap\..*[0-9]{4}-[0-9]{2}-[0-9]{2}', text)
        cap_number = cap_match.group(0)
        print(f"This is Cap. {cap_number}")
        # Extract first sentence
        # Looking for text after the date until the first period
        first_sentence_pattern = r'\d{4}-\d{2}-\d{2}(.*?(?:[.:]|\S\s{2}))(.*)'
        first_sentence_match = re.search(first_sentence_pattern, text)

        if first_sentence_match:
            first_sentence = first_sentence_match.group(1).strip()
            remaining_text = first_sentence_match.group(2).strip()
        else:
            first_sentence = None
            remaining_text = None


        # 检查是否是附属法例
        if 'sub. leg.' in cap_number.lower():
            # 添加到主法例的sub列表中
            sub_law = {
                "sub_title": cap_number,
                "details": [remaining_text],
                "first sentence": first_sentence
            }

            laws[-1]["sub"].append(sub_law)
            continue
        else:
            # 如果是主法例，创建新的主法例记录
            current_law = {
                "title": cap_number,
                "first sentence": first_sentence,
                "details": [remaining_text],
                "sub": [],
            }
            laws.append(current_law)

    return laws

def save_laws(laws, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(laws, f, ensure_ascii=False, indent=2)

def main():
    input_file = "./law_data/hkel_en.jsonl"
    output_file = "./law_data/structured_laws.json"

    print("Starting law file processing...")
    laws = parse_law_file(input_file)
    print(f"Found {len(laws)} laws")

    save_laws(laws, output_file)
    print(f"Successfully saved structured data to {output_file}")


if __name__ == "__main__":
    main()