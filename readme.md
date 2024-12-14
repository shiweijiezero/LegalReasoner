# 环境配置
```bash
export HUGGINGFACE_HUB_CACHE="/aifs4su/hansirui/HK/huggingface/"
export TRANSFORMERS_CACHE="/aifs4su/hansirui/HK/huggingface/"
export HF_HOME="/aifs4su/hansirui/HK/huggingface/"
export HF_TOKEN=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
```

# 数据处理
0. 抽取法条文本
```bash
python hk_law_parser.py
```
1. 重命名，去除文件名中的空格
```bash
python rename_files.py
```
2. 抽取word文档中的文本，保存为.txt
```bash
python data_processing.py
```
3. 对于短的内容，先抽取内容
- 起诉方、被告方
- 诉讼请求
- 事实
- 争议点(如有)
- 法院意见
- 判决结果


# 法条定位
## 爬虫
```bash
cd law_data
python spider.py
python spider-pool.py
```


## 法条结构
- Cap. 1
  - Long Title
  - Part I
    - Section 1
      - Content/Text/longTitle (Optional)
      - Article 1 (Optional)
        - Subsection 1 (Optional)
          - leadin (Optional)
          - Paragraph (Optional)
            - leadin (Optional)
            - Subparagraph (Optional)
## 构造数据
首先需要定位法条的Cap. 这样首先需要一系列的Question，随后针对每个Question，利用开源模型去遍历询问相关性，最后得到一个相关法条集合
## 微调Qwen小模型用于法条定位
添加一系列(1000余个) special token，用于表示法条的Cap. 通过微调Qwen小模型，使其能够定位到相关的法条
## 二分类
候选方法，使用Qwen小模型/CoBert模型来进行二分类，为了提升效率，可以一次性对包含多个法条的文本进行判断
也可以建模为index模型，一次性输入多个法条，输出对应的index 或者 -1