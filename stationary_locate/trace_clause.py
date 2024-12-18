from pprint import pprint

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification

import clause_utils

def main():
    hierarchy_law = clause_utils.build_hierarchy("law_data/spider")
    caps = hierarchy_law.get_caps()
    caps_list = [f"{cap['lower_cap_number']} {cap['lower_title']}: {cap['lower_long_title']}" for key, cap in caps.items()]
    # pprint(caps_list)

    # 初始化tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained('./bert_results/')
    model = AutoModelForSequenceClassification.from_pretrained(
        './bert_results/',
        num_labels=2,  #
        # problem_type="single_label_classification",
        torch_dtype=torch.bfloat16,

    )
    # model.to("cuda")

    while True:
        text = input("请输入事实：")
        if text == "exit":
            break
        model_input = f"{text[:2000]} [SEP]"
        model_inputs = [f"{model_input} {cap}" for cap in caps_list]
        inputs = tokenizer(model_inputs, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        # 解析出cap number和title
        # Extract logits and get predictions
        logits = outputs.logits  # Shape: (num_caps, num_labels)
        predictions = torch.argmax(logits, dim=1)  # Get the predicted label for each input pair

        # Filter and display relevant caps
        relevant_caps = [(caps_list[i], predictions[i].item()) for i in range(len(predictions)) if predictions[i] == 1]
        if relevant_caps:
            for cap, pred in relevant_caps:
                cap_number, title = cap.split(' ', 1)  # Split at the first space to separate number and title
                print(f"Relevant Cap: {cap_number} - {title}")
        else:
            print("No relevant caps found for the input.")

if __name__ == "__main__":
    main()


# On the morning of 24 August 2013, the defendant arrived in Hong Kong by air and was attempting to use a special immigration channel.\nThe defendant was told he could not use this channel and became upset. He also smelt of alcohol.\nAn immigration officer directed him towards normal clearance, and the defendant's behavior attracted the attention of three police officers who were on anti-terrorist duty.\nShortly after that, the defendant allegedly grabbed one of the sub-machine guns held by one of the police officers and pulled it and the officer towards him. The defendant was subdued and apologised.\nPW1, PW2, PW3, PW4, and PW5 witnessed the incident. PW3's evidence was that he saw the defendant gra bbing and pulling his sub-machine gun.\nA CCTV camera recorded the area, capturing the incident, though it was obscured at times.\nThe defendant’s wife and 12-year-old daughter went through immigration ahead of him, and he had travelled from Incheon International Airport with them.\nThe defendant described the incident as an attempt to apologize to the police officer, citing cultural norms as his explanation for the alleged grabbing action.\nPW3, the police officer, verified the defendant's alleged alcohol intoxication. [SEP] CAP 69 Private Bills Ordinance