# 层级关系为
# - Cap. 1
#     - Long Title
#     - Part I
#         - Section 1
#             - Content/Text/longTitle (Optional)
#             - Article 1 (Optional)
#                 - Subsection 1 (Optional)
#                     - leadin (Optional)
#                     - Paragraph (Optional)
#                         - leadin (Optional)
#                         - Subparagraph (Optional)
import json
import os
import random
from pprint import pprint
from tqdm import tqdm

class Hierarchy:
    def __init__(self):
        self.caps = {}

    def add_cap(self, cap):
        self.caps[cap.cap_number] = cap

    def print(self, indent=0):
        for cap in self.caps.values():
            print("  " * indent + f"Cap. {cap.cap_number}: {cap.title}")
            cap.print(indent + 1)

    def get_caps(self):
        caps = {}
        for cap_number, cap_data in self.caps.items():
            caps[cap_number] = {
                'cap_number': cap_number,  # 保持原始大写
                'lower_cap_number': cap_number.lower(),  # 用于匹配
                'title': cap_data.title,   # 保持原始大写
                'lower_title': cap_data.title.lower(),  # 用于匹配
                'long_title': cap_data.parts[0].content if 0 in cap_data.parts else None,
                'lower_long_title': cap_data.parts[0].content.lower() if 0 in cap_data.parts else None
            }
        return caps

class Cap:
    def __init__(self, cap_number, title):
        self.cap_number = cap_number
        self.title = title
        self.parts = {}

    def add_part(self, part):
        self.parts[part.part_number] = part

    def print(self, indent=0):
        for part in self.parts.values():
            if part.part_number == 0 and not part.sections:
                continue  # 如果默认part没有内容则不打印
            print("  " * indent + f"Part {part.part_number}: {part.title}")
            if part.content:
                print("  " * (indent + 1) + f"Content: {part.content}")
            part.print(indent + 1)

    def get_part_section_table_of_contents(self):
        toc = []
        toc.append(f"Cap. {self.cap_number}: {self.title}")
        for part_idx, part in self.parts.items():
            toc.append(f"Part {part.part_number}: {part.title}")
            for section_idx, section in part.sections.items():
                toc.append(f"Section {section.section_number}: {section.title}")
        return toc

class Part:
    def __init__(self, part_number, title, content=None):
        self.part_number = part_number
        self.title = title
        self.content = content
        self.sections = {}

    def add_section(self, section):
        self.sections[section.section_number] = section

    def print(self, indent=0):
        for section in self.sections.values():
            print("  " * indent + f"Section {section.section_number}: {section.title}")
            section.print(indent + 1)

class Section:
    def __init__(self, section_number, title):
        self.section_number = section_number
        self.title = title
        self.content = None
        self.articles = []

    def add_article(self, article):
        self.articles.append(article)

    def print(self, indent=0):
        if self.content:
            print("  " * indent + f"Content: {self.content}")
        for article in self.articles:
            print("  " * indent + f"Article: {article.heading}")
            article.print(indent + 1)

    def get_section_toc(self, prefix=""):
        toc = []
        if prefix:
            toc.append(f"{prefix}")
        toc.append(f"Section {self.section_number}: {self.title}")
        if self.content:
            toc.append(f"Section Content: {self.content}")
        for article in self.articles:
            toc.append(f"- Article: {article.heading}")
            for subsection in article.subsections:
                toc.append(f"-- Subsection {subsection.subsection_number}")
                if subsection.leadin:
                    toc.append(f"-- Subsection Leadin: {subsection.leadin}")
                if subsection.content:
                    toc.append(f"-- Subsection Content: {subsection.content}")
                for paragraph in subsection.paragraphs:
                    toc.append(f"--- Paragraph {paragraph.paragraph_number}")
                    if paragraph.leadin:
                        toc.append(f"--- Paragraph Leadin: {paragraph.leadin}")
                    if paragraph.content:
                        toc.append(f"--- Paragraph Content: {paragraph.content}")
                    for subparagraph in paragraph.subparagraphs:
                        toc.append(f"---- Subparagraph {subparagraph.number}")
                        toc.append(f"---- Subparagraph Content: {subparagraph.content}")
        toc = "\n".join(toc)
        return toc

class Article:
    def __init__(self, heading):
        self.heading = heading
        self.subsections = []

    def add_subsection(self, subsection):
        self.subsections.append(subsection)

    def print(self, indent=0):
        for subsection in self.subsections:
            print("  " * indent + f"Subsection {subsection.subsection_number}")
            subsection.print(indent + 1)

class Subsection:
    def __init__(self, subsection_number, leadin=None):
        self.subsection_number = subsection_number
        self.leadin = leadin
        self.content = None
        self.paragraphs = []

    def add_paragraph(self, paragraph):
        self.paragraphs.append(paragraph)

    def print(self, indent=0):
        if self.leadin:
            print("  " * indent + f"Leadin: {self.leadin}")
        if self.content:
            print("  " * indent + f"Content: {self.content}")
        for paragraph in self.paragraphs:
            print("  " * indent + f"Paragraph {paragraph.paragraph_number}")
            paragraph.print(indent + 1)

class Paragraph:
    def __init__(self, paragraph_number, leadin=None):
        self.paragraph_number = paragraph_number
        self.leadin = leadin
        self.content = None
        self.subparagraphs = []

    def add_subparagraph(self, subparagraph):
        self.subparagraphs.append(subparagraph)

    def print(self, indent=0):
        if self.leadin:
            print("  " * indent + f"Leadin: {self.leadin}")
        if self.content:
            print("  " * indent + f"Content: {self.content}")
        for subparagraph in self.subparagraphs:
            print("  " * indent + f"Subparagraph {subparagraph.number}: {subparagraph.content}")

class Subparagraph:
    def __init__(self, number, content):
        self.number = number
        self.content = content

def parse_content_structure(section_data):
    """递归解析内容结构"""
    # meta_data = section_data.get('metadata', {})
    content_data = section_data.get('content', {})
    if not content_data:
        return None

    section_number = content_data.get('number', '').strip('.')
    section = Section(section_number, content_data.get('heading', ''))
    section.content = content_data.get('content')

    # 处理 articles
    if 'articles' in content_data:
        for article_data in content_data['articles']:
            # 如果不是default article，则清空content
            section.content = None  # 清空content

            article = Article(article_data.get('heading', ''))

            # 处理 article 下的 subsections
            for subsection_data in article_data.get('subsections', []):
                subsection = create_subsection(subsection_data)
                article.add_subsection(subsection)

            section.add_article(article)

    # 如果有直接的subsections，创建一个默认article
    elif 'subsections' in content_data and content_data['subsections']:
        default_article = Article("Default Article")
        for subsection_data in content_data['subsections']:
            subsection = create_subsection(subsection_data)
            default_article.add_subsection(subsection)
        section.add_article(default_article)

    return section

def create_subsection(subsection_data):
    """创建 subsection 及其子结构"""
    subsection = Subsection(
        subsection_data.get('number', '').strip('.'),
        subsection_data.get('leadin')
    )
    subsection.content = subsection_data.get('content')

    # 处理 paragraphs
    if 'paragraphs' in subsection_data:
        for para_data in subsection_data['paragraphs']:
            # 清空content
            subsection.content = None
            paragraph = Paragraph(
                para_data.get('number', '').strip('.'),
                para_data.get('leadin')
            )
            paragraph.content = para_data.get('content')

            # 处理 subparagraphs
            if 'subparagraphs' in para_data:
                for subpara_data in para_data['subparagraphs']:
                    # 清空content
                    paragraph.content = None
                    subparagraph = Subparagraph(
                        subpara_data.get('number', '').strip('.'),
                        subpara_data.get('content')
                    )
                    paragraph.add_subparagraph(subparagraph)

            subsection.add_paragraph(paragraph)

    return subsection

def parse_section_file(file_path):
    """解析单个section文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {
        'type': data.get('type', ''),
        'metadata': data.get('metadata', {}),
        'content': data.get('content', {})
    }

def build_hierarchy(base_path):
    """构建完整的层级结构"""
    hierarchy = Hierarchy()

    paths = os.listdir(base_path)
    for path in tqdm(paths, desc='Building hierarchy'):
        full_path = os.path.join(base_path, path)
        if not os.path.isdir(full_path):
            continue

        # 读取info.json
        info_path = os.path.join(full_path, 'info.json')
        if not os.path.exists(info_path):
            continue

        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        # 创建Cap对象
        cap = Cap(info['cap_number'], info['title'])
        hierarchy.add_cap(cap)

        # 读取所有section文件
        current_part = None
        files = [f for f in os.listdir(full_path)
                 if f.startswith('section_') and f.endswith('.json')]
        files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        for file in files:
            file_path = os.path.join(full_path, file)
            section_data = parse_section_file(file_path)

            if section_data['type'] == 'part':
                # 创建新的Part
                part_number = len(cap.parts)
                title = section_data['metadata']['section_title']
                content = section_data['content'].get('text', '')
                current_part = Part(part_number, title, content)
                cap.add_part(current_part)

            elif section_data['type'] == 'section':
                # 解析完整的section结构
                section = parse_content_structure(section_data)
                if section:
                    # 如果没有当前part，使用默认part
                    target_part = current_part if current_part else Part(0, "Default Part")
                    target_part.add_section(section)

            elif section_data['type'] == 'longtitle':
                # 处理长标题
                # 创建longtitle part
                current_part = Part(0, "Long Title")
                cap.add_part(current_part)
                if current_part:
                    current_part.title = section_data['metadata']['section_title']
                    if 'content' in section_data['content']:
                        current_part.content = section_data['content']['content']
                # 初始化current_part为默认part
                current_part = Part(0, "Default Part")

        # hierarchy.print()
        # exit()
    return hierarchy

from typing import List, Dict, Optional, Tuple
import re

class LawMatcher:
    def __init__(self, hierarchy_law):
        self.hierarchy_law = hierarchy_law
        # 预处理 caps 数据，转换为小写
        self.processed_caps = {}
        for cap_number, cap_data in hierarchy_law.caps.items():
            self.processed_caps[cap_number] = {
                'cap_number': cap_number,  # 保持原始大写
                'title': cap_data.title,   # 保持原始大写
                'lower_title': cap_data.title.lower(),  # 用于匹配
                'lower_long_title': cap_data.parts[0].content.lower() if 0 in cap_data.parts else None
            }

    def get_negative_laws(self, laws, number):
        negative_laws = []
        law_numbers= [law['cap_number'] for law in laws]
        while len(negative_laws) < number:
            cap = random.choice(list(self.processed_caps.values()))
            if cap['cap_number'] not in law_numbers:
                negative_laws.append(cap)
        return negative_laws


    def clean_law_text(self, law_text: str) -> str:
        """清理单个法条文本"""
        if not law_text:
            return ""
        return law_text.lower().strip()

    def match_cap_number(self, law_text: str) -> Optional[Tuple[str, str, str]]:
        """匹配CAP号，返回(cap_number, title)"""
        pattern = r'cap\s+(\d+)'
        matches = re.findall(pattern, law_text)

        for match in matches:
            cap_str = f"CAP {match}"
            for cap_number, cap_data in self.processed_caps.items():
                if cap_str.lower() == cap_number.lower():
                    return cap_number, cap_data['title'], cap_data['lower_long_title']
        return None

    def match_title(self, law_text: str) -> Optional[Tuple[str, str, str]]:
        """匹配法规标题，返回(cap_number, title)"""
        for cap_number, cap_data in self.processed_caps.items():
            if cap_data['lower_title'].lower() in law_text.lower():
                return cap_number, cap_data['title'], cap_data['lower_long_title']
        return None

    def validate_law(self, law_text: str) -> Optional[Tuple[str, str, str]]:
        """判断一条法规是否有效，返回(cap_number, title)或None"""
        if type(law_text) != str:
            law_text = str(law_text)
        law_text = self.clean_law_text(law_text)

        # 优先匹配CAP号
        cap_match = self.match_cap_number(law_text)
        if cap_match:
            return cap_match

        # 其次匹配标题
        title_match = self.match_title(law_text)
        if title_match:
            return title_match

        return None

    def process_data(self, data: List[dict]) -> List[dict]:
        """处理整个数据集，返回有效的数据条目"""
        valid_data = []

        for item in tqdm(data, desc='Processing data', total=len(data)):
            related_laws = item.get('related_laws', [])
            if not isinstance(related_laws, list):
                continue
            facts = item.get('facts', '')
            valid_laws = []
            for law in related_laws:
                match_result = self.validate_law(law)
                if match_result:
                    cap_number, title, lower_long_title = match_result
                    valid_laws.append({
                        'cap_number': cap_number,
                        'title': title,
                        'lower_long_title': lower_long_title
                    })

            if valid_laws:
                # 添加处理后的法条信息到原数据中
                item['processed_laws'] = valid_laws
                valid_data.append(item)
            else:
                item['processed_laws'] = []
                valid_data.append(item)
        return valid_data


if __name__ == "__main__":
    hierarchy = build_hierarchy("law_data/spider")
    hierarchy.print()