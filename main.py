import json


def main():
    with open("law_data/structured_laws.json", "r", encoding="utf-8") as f:
        laws = json.load(f)

    with open("processed_data/analysis_results.json", "r", encoding="utf-8") as f:
        analysis_results = json.load(f)




if __name__ == "__main__":
    main()