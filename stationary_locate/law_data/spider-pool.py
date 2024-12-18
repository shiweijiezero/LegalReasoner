from pprint import pprint
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import os
import utils
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import logging
from datetime import datetime

# 设置日志
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/scraper_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    return webdriver.Chrome(options=chrome_options)

def save_section_content(section_data, filename: str):
    """Save the structured section content to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(section_data, f, ensure_ascii=False, indent=4)

def scrape_ordinance(driver, cap_no, max_retries=3):
    """获取特定法例的标题、编号和所有链接"""
    if os.path.exists(f"spider/cap_{cap_no}/info.json"):
        logging.info(f"Cap {cap_no} already scraped")
        with open(f"spider/cap_{cap_no}/info.json", "r", encoding="utf-8") as f:
            ordinance_info = json.load(f)
        with open(f"spider/cap_{cap_no}/links.json", "r", encoding="utf-8") as f:
            links = json.load(f)
        return links, ordinance_info

    retry_count = 0
    while retry_count < max_retries:
        sleep_time = 0.2 + retry_count * 1.5
        try:
            url = f"https://www.hklii.hk/en/legis/ord/{cap_no}"
            driver.get(url)
            time.sleep(sleep_time)

            cap_number = driver.find_element(By.CLASS_NAME, "titlecapno").text.strip()
            title = driver.find_element(By.CLASS_NAME, "text-h2").text.strip()
            ordinance_info = {
                "cap_number": cap_number,
                "title": title
            }

            elements = driver.find_elements(By.CSS_SELECTOR, "td.toctitle a")
            links = [element.get_attribute('href') for element in elements if element.get_attribute('href') != url]

            if not links:
                raise Exception("No links found")

            cap_dir = f"spider/cap_{cap_no}"
            os.makedirs(cap_dir, exist_ok=True)

            with open(f"{cap_dir}/info.json", "w", encoding="utf-8") as f:
                json.dump(ordinance_info, f, indent=4, ensure_ascii=False)

            with open(f"{cap_dir}/links.json", "w", encoding="utf-8") as f:
                json.dump(links, f, indent=4)

            logging.info(f"Successfully scraped Cap {cap_no} on attempt {retry_count + 1}")
            return links, ordinance_info

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                logging.warning(f"Attempt {retry_count} failed for Cap {cap_no}: {e}")
                logging.info(f"Retrying... ({max_retries - retry_count} attempts remaining)")
                time.sleep(retry_count * 2)
            else:
                logging.error(f"Failed to scrape Cap {cap_no} after {max_retries} attempts: {e}")
                return [], None

def process_ordinance_content(driver, cap_no, links, ordinance_info):
    """处理单个法例的所有内容"""
    cap_dir = f"spider/cap_{cap_no}"
    contents = []
    for i, link in enumerate(links):
        filename = f"{cap_dir}/section_{i + 1}.json"
        if os.path.exists(filename):
            continue

        content = utils.get_section_content(driver, link, ordinance_info)
        if content:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=4, ensure_ascii=False)
            logging.info(f"Saved {filename}")
        contents.append(content)
    return contents

def process_single_cap(cap_no):
    """处理单个cap的函数，作为进程的目标函数"""
    try:
        driver = setup_driver()
        logging.info(f"Processing Cap {cap_no}...")

        try:
            links, ordinance_info = scrape_ordinance(driver, cap_no)
            if not links:
                return

            logging.info(f"Found {len(links)} sections for Cap {cap_no}")
            contents = process_ordinance_content(driver, cap_no, links, ordinance_info)

        except Exception as e:
            # logging.error(f"Error processing Cap {cap_no}: {e}")
            pass

    except Exception as e:
        logging.error(f"Fatal error in process for Cap {cap_no}: {e}")
    finally:
        if driver:
            driver.quit()

def chunk_ranges(ranges, num_chunks):
    """将范围分成大致相等的几块"""
    avg = len(ranges) // num_chunks
    remainder = len(ranges) % num_chunks
    result = []
    start = 0
    for i in range(num_chunks):
        end = start + avg + (1 if i < remainder else 0)
        result.append(ranges[start:end])
        start = end
    return result

def main():
    setup_logging()
    ranges = list(range(1, 14)) + list(range(16, 651)) + list(range(1001, 1183))
    # ranges = list(range(282, 651)) + list(range(1001, 1183))

    # 获取CPU核心数，保留一个核心给系统
    # num_processes = max(1, multiprocessing.cpu_count() - 1)
    num_processes = 15
    logging.info(f"Starting scraping with {num_processes} processes")

    # 将范围分成几块，每个进程处理一块
    chunked_ranges = chunk_ranges(ranges, num_processes)

    # 创建进程池
    with Pool(num_processes) as pool:
        # for chunk in tqdm(chunked_ranges, desc="Processing chunks", total=len(chunked_ranges)):
            # 对每个chunk中的cap_no启动一个进程
        pool.map(process_single_cap, ranges)

if __name__ == "__main__":
    main()