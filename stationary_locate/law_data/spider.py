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
    """
    获取特定法例的标题、编号和所有链接

    Args:
        driver: Selenium WebDriver实例
        cap_no: 法例编号
        max_retries: 最大重试次数，默认3次

    Returns:
        tuple: (links列表, ordinance_info字典)
    """
    # 检查是否已经爬取过
    if os.path.exists(f"spider/cap_{cap_no}/info.json"):
        print(f"Cap {cap_no} already scraped")
        # 读取标题信息
        with open(f"spider/cap_{cap_no}/info.json", "r", encoding="utf-8") as f:
            ordinance_info = json.load(f)

        # 读取链接
        with open(f"spider/cap_{cap_no}/links.json", "r", encoding="utf-8") as f:
            links = json.load(f)
        return links, ordinance_info

    retry_count = 0
    while retry_count < max_retries:
        sleep_time = 0.2+retry_count * 1.5
        try:
            url = f"https://www.hklii.hk/en/legis/ord/{cap_no}"
            driver.get(url)
            time.sleep(sleep_time)  # 考虑使用显式等待替代固定休眠

            # 抽取标题和编号
            cap_number = driver.find_element(By.CLASS_NAME, "titlecapno").text.strip()
            title = driver.find_element(By.CLASS_NAME, "text-h2").text.strip()
            ordinance_info = {
                "cap_number": cap_number,
                "title": title
            }

            # 获取所有链接
            elements = driver.find_elements(By.CSS_SELECTOR, "td.toctitle a")
            links = [element.get_attribute('href') for element in elements if element.get_attribute('href') != url]

            if not links:  # 如果没有找到任何链接，可能是页面加载问题
                raise Exception("No links found")

            # 创建保存目录
            cap_dir = f"spider/cap_{cap_no}"
            os.makedirs(cap_dir, exist_ok=True)

            # 保存标题信息
            with open(f"{cap_dir}/info.json", "w", encoding="utf-8") as f:
                json.dump(ordinance_info, f, indent=4, ensure_ascii=False)

            # 保存链接
            with open(f"{cap_dir}/links.json", "w", encoding="utf-8") as f:
                json.dump(links, f, indent=4)

            print(f"Successfully scraped Cap {cap_no} on attempt {retry_count + 1}")
            return links, ordinance_info

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"Attempt {retry_count} failed for Cap {cap_no}: {e}")
                print(f"Retrying... ({max_retries - retry_count} attempts remaining)")
                time.sleep(retry_count * 2)  # 递增等待时间
            else:
                print(f"Failed to scrape Cap {cap_no} after {max_retries} attempts: {e}")
                return [], None


def process_ordinance_content(driver, cap_no, links, ordinance_info):
    """处理单个法例的所有内容"""

    cap_dir = f"spider/cap_{cap_no}"
    contents = []
    for i, link in tqdm(enumerate(links), desc=f"Processing Cap {cap_no}", total=len(links)):
        # 检查是否已经爬取过
        try:
            filename = f"{cap_dir}/section_{i + 1}.json"
            if os.path.exists(filename):
                continue

            content = utils.get_section_content(driver, link, ordinance_info)
            if content:
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=4, ensure_ascii=False)
                print(f"Saved {filename}")
            # pprint(content)
            # if i==5:
            #     exit()
            contents.append(content)
        except Exception as e:
            print(f"Error processing section {i + 1} for Cap {cap_no}: {e}")
            continue
    return contents



def main():
    ranges = list(range(1,14))+ list(range(16,651)) + list(range(1001, 1183))
    # ranges = list(range(485,486))
    driver = None

    try:
        driver = setup_driver()
        # links, ordinance_info = scrape_ordinance(driver, 2)
        # exit()
        for cap_no in tqdm(ranges):
            print(f"\nProcessing Cap {cap_no}...")
            # 获取链接
            try:
                links, ordinance_info = scrape_ordinance(driver, cap_no)
                print(ordinance_info)
                if not links:
                    continue
                print(f"Found {len(links)} sections")

                # 处理内容
                contents = process_ordinance_content(driver, cap_no, links, ordinance_info)

            except Exception as e:
                print(f"Error processing Cap {cap_no}: {e}")
                continue


    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if driver:
            driver.quit()


if __name__ == "__main__":
    main()
