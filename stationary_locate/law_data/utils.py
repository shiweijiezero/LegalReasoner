from pprint import pprint

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from typing import Optional, Dict, Any, List
import json
import re


def get_section_content(driver, link: str, ordinance_info, retries: int = 3):
    """
    Extract legal section content with proper structure.

    Args:
        driver: Selenium WebDriver instance
        link: URL of the section page
        retries: Number of retry attempts

    Returns:
        Dictionary containing structured section content or None if extraction fails
    """
    for attempt in range(retries):
        # try:
        driver.get(link)
        time.sleep(0.5)  # Allow page load

        # Wait for main content
        content_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "case-content"))
        )

        # Extract metadata
        metadata = _extract_metadata(driver)

        # Extract main content based on content type
        content_type = _determine_content_type(content_div)
        main_content = _extract_main_content(content_div, content_type)

        # Build complete section data
        section_data = {
            "ordinance_info": ordinance_info,
            "metadata": metadata,
            "type": content_type,
            "content": main_content
        }

        # Add source notes if present
        source_notes = _extract_source_notes(content_div)
        if source_notes:
            section_data["source_notes"] = source_notes

        return section_data

    return None


def _extract_metadata(driver) -> Dict[str, str]:
    """Extract section metadata"""
    try:
        # Get basic section info
        section_number = ""
        section_title = ""

        title_elements = driver.find_elements(By.CLASS_NAME, "titlecapno")
        if title_elements:
            section_number = title_elements[0].text.strip()

        subtitle_elements = driver.find_elements(By.CLASS_NAME, "text-h4")
        if subtitle_elements:
            section_title = subtitle_elements[0].text.strip()

        # Get version info
        version_date = ""
        date_elements = driver.find_elements(By.CLASS_NAME, "spec_infoval")
        for elem in date_elements:
            if "Latest Version" in elem.text:
                version_date = elem.text.replace("(Latest Version)", "").strip()
                break

        return {
            "section_number": section_number,
            "section_title": section_title,
            "version_date": version_date
        }
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}


def _determine_content_type(content_div) -> str:
    """Determine the type of legal content"""
    content_tags = ["longtitle", "section", "schedule", "part"]
    for tag in content_tags:
        if content_div.find_elements(By.TAG_NAME, tag):
            return tag
    return "unknown"


def _extract_main_content(content_div, content_type: str) -> Dict[str, Any]:
    """Extract main content based on content type"""
    if content_type == "longtitle":
        return _extract_longtitle_content(content_div)
    elif content_type == "section":
        return _extract_section_content(content_div)
    else:
        # Extract text content for other types
        return {
            "text": content_div.text.strip()
        }


def _extract_longtitle_content(content_div) -> Dict[str, str]:
    """Extract content from longtitle section"""
    longtitle = content_div.find_element(By.TAG_NAME, "longtitle")
    content = longtitle.find_element(By.TAG_NAME, "content").text.strip()

    # Extract commencement note if present
    commencement = ""
    try:
        commencement = content_div.find_element(By.TAG_NAME, "commencementnote").text.strip()
    except:
        pass

    return {
        "content": content,
        "commencement": commencement
    }


def _extract_section_content(content_div) -> Dict[str, Any]:
    """Extract content from regular section with support for complex hierarchical structure"""
    section = content_div.find_element(By.TAG_NAME, "section")

    # Get section attributes
    section_data = {
        "id": section.get_attribute("id"),
        "name": section.get_attribute("name"),
        "status": section.get_attribute("status"),
        "reason": section.get_attribute("reason"),
        "startperiod": section.get_attribute("startperiod"),
        "temporalid": section.get_attribute("temporalid"),
        "subsections": []
    }

    # Extract section number if present
    try:
        num_elem = section.find_element(By.TAG_NAME, "num")
        section_data["number"] = num_elem.text.strip()
    except:
        section_data["number"] = ""

    # Extract heading if present
    try:
        heading_elem = section.find_element(By.TAG_NAME, "heading")
        section_data["heading"] = heading_elem.text.strip()
    except:
        section_data["heading"] = ""

    # Extract any direct content if present (for simple sections)
    try:
        content_elem = section.find_element(By.TAG_NAME, "content")
        section_data["content"] = content_elem.text.strip()
    except:
        section_data["content"] = ""

    try:
        content_elem = section.find_element(By.TAG_NAME, "text")
        section_data["content"] += content_elem.text.strip()
    except:
        section_data["content"] += ""

    # Extract Article if present
    # Get all direct children of the section
    children = section.find_elements(By.XPATH, "./*")
    flag = None
    for element in children:
        tag_name = element.tag_name
        if tag_name == "crossheading":
            flag = "crossheading"

    if flag == "crossheading":
        children = section.find_elements(By.XPATH, "./*")
        current_heading = ""
        current_subsections = []
        cross_headings = []

        for idx, element in enumerate(children):
            tag_name = element.tag_name
            if tag_name == "crossheading":
                if current_subsections and idx > 1:
                    cross_headings.append({
                        "heading": current_heading,
                        "subsections": current_subsections
                    })
                    current_subsections = []
                heading = element.text.strip()
                current_heading += heading

            if tag_name == "subsection":
                subsection_data = _extract_subsection(element)
                current_subsections.append(subsection_data)

        if current_subsections:
            cross_headings.append({
                "heading": current_heading,
                "subsections": current_subsections
            })

        section_data["articles"] = cross_headings

    else:
        # Extract subsections
        try:
            subsections = section.find_elements(By.TAG_NAME, "subsection")
            for subsection in subsections:
                subsection_data = _extract_subsection(subsection)
                section_data["subsections"].append(subsection_data)
        except:
            pass

    # Extract source notes if present
    try:
        sourcenote = section.find_element(By.TAG_NAME, "sourcenote")
        section_data["sourcenote"] = sourcenote.text.strip()
    except:
        section_data["sourcenote"] = ""

    return section_data


def _extract_subsection(subsection) -> Dict[str, Any]:
    """Extract content from a subsection including paragraphs"""
    subsection_data = {
        "id": subsection.get_attribute("id"),
        "name": subsection.get_attribute("name"),
        "temporalid": subsection.get_attribute("temporalid"),
        "number": "",
        "content": "",
        "paragraphs": []
    }

    # Get subsection number
    try:
        num_elem = subsection.find_element(By.TAG_NAME, "num")
        subsection_data["number"] = num_elem.text.strip()
    except:
        pass

    # Get lead-in text if present
    try:
        leadin_elem = subsection.find_element(By.TAG_NAME, "leadin")
        subsection_data["leadin"] = leadin_elem.text.strip()
    except:
        subsection_data["leadin"] = ""

    # Get main content if present
    try:
        content_elem = subsection.find_element(By.TAG_NAME, "content")
        subsection_data["content"] = content_elem.text.strip()
    except:
        pass

    # Get paragraphs
    try:
        paragraphs = subsection.find_elements(By.TAG_NAME, "paragraph")
        for para in paragraphs:
            para_data = _extract_paragraph(para)
            subsection_data["paragraphs"].append(para_data)
    except:
        pass

    return subsection_data


def _extract_paragraph(para) -> Dict[str, Any]:
    """Extract content from a paragraph including subparagraphs"""
    para_data = {
        "id": para.get_attribute("id"),
        "name": para.get_attribute("name"),
        "temporalid": para.get_attribute("temporalid"),
        "number": "",
        "content": "",
        "leadin": "",
        "subparagraphs": []
    }

    # Get paragraph number/letter
    try:
        num_elem = para.find_element(By.TAG_NAME, "num")
        para_data["number"] = num_elem.text.strip()
    except:
        pass

    # Get lead-in text if present
    try:
        leadin_elem = para.find_element(By.TAG_NAME, "leadin")
        para_data["leadin"] = leadin_elem.text.strip()
    except:
        pass

    # Get content if present
    try:
        content_elem = para.find_element(By.TAG_NAME, "content")
        para_data["content"] = content_elem.text.strip()
    except:
        pass

    # Get subparagraphs
    try:
        subparas = para.find_elements(By.TAG_NAME, "subparagraph")
        for subpara in subparas:
            subpara_data = _extract_subparagraph(subpara)
            para_data["subparagraphs"].append(subpara_data)
    except:
        pass

    return para_data


def _extract_subparagraph(subpara) -> Dict[str, Any]:
    """Extract content from a subparagraph"""
    return {
        "id": subpara.get_attribute("id"),
        "name": subpara.get_attribute("name"),
        "temporalid": subpara.get_attribute("temporalid"),
        "number": subpara.find_element(By.TAG_NAME, "num").text.strip(),
        "content": subpara.find_element(By.TAG_NAME, "content").text.strip()
    }


def _extract_source_notes(content_div) -> str:
    """Extract source notes if present"""
    try:
        return content_div.find_element(By.TAG_NAME, "sourcenote").text.strip()
    except:
        return ""
