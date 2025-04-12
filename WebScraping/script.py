import requests
from bs4 import BeautifulSoup
import time
import json
import csv
import random
import os

BASE_URL = "https://www.nhs.uk"
START_URL = f"{BASE_URL}/conditions/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

START_INDEX = 4
MAX_CONDITIONS = 10

def get_condition_links():
    response = requests.get(START_URL, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    panels = soup.find_all("ul", class_="nhsuk-list nhsuk-list--border")
    links = []

    for panel in panels:
        for a in panel.find_all("a", href=True):
            href = a['href']
            if href.startswith("/conditions/"):
                full_link = BASE_URL + href
                if full_link not in links:
                    links.append(full_link)

    return links[:MAX_CONDITIONS]

def extract_description(soup):
    desc_paragraphs = []
    main_container = soup.find("div", class_="nhsuk-width-container")
    if main_container:
        for p in main_container.find_all("p", recursive=True):
            text = p.get_text(strip=True)
            if text and len(text.split()) > 4:
                desc_paragraphs.append(text)
            if len(desc_paragraphs) >= 2:
                break
    if desc_paragraphs:
        return "\n".join(desc_paragraphs).strip()

    h1 = soup.find("h1")
    if h1:
        for sibling in h1.find_next_siblings():
            if sibling.name == "p":
                text = sibling.get_text(strip=True)
                if len(text.split()) > 4:
                    return text.strip()
    return "N/A"

def extract_symptoms(soup):
    content = []
    possible_titles = [
        "symptoms",
        "signs and symptoms",
        "symptoms of",
        "what are the symptoms"
    ]

    headings = soup.find_all(["h2", "h3"])
    for heading in headings:
        heading_text = heading.get_text(strip=True).lower()
        if any(phrase in heading_text for phrase in possible_titles):
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3"]:
                    break
                if sibling.name == "p":
                    content.append(sibling.get_text(strip=True))
                if sibling.name == "ul":
                    for li in sibling.find_all("li"):
                        content.append("• " + li.get_text(strip=True))
            break

    return "\n".join(content).strip() or "N/A"

def scrape_condition(url):
    print(f" Scraping: {url}")
    try:
        time.sleep(random.uniform(5.5, 10.5))
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("h1").text.strip()
    except Exception as e:
        print(f" Error at title: {e}")
        return None

    return {
        "disease": title,
        "description": extract_description(soup),
        "symptoms": extract_symptoms(soup)
    }


def load_existing_data(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


all_data = []

print(" Extracting links...")
links = get_condition_links()
print(f"Found {len(links)} links.")

links_to_scrape = links[START_INDEX:]

for idx, link in enumerate(links_to_scrape, start=START_INDEX + 1):
    try:
        data = scrape_condition(link)
        if data:
            all_data.append(data)
        print(f" {idx}/{len(links)} complete.")
        time.sleep(random.uniform(2, 8))
    except Exception as e:
        print(f" Erorr at {link}: {e}")

print(f"\n Diseases length {len(all_data)}")


existing_data = load_existing_data("boli_nhs.json")
combined_data = existing_data + all_data

seen = set()
final_data = []
for item in combined_data:
    name = item.get("disease", "").strip().lower()
    if name and name not in seen:
        final_data.append(item)
        seen.add(name)


with open("boli_nhs.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)


with open("boli_nhs.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["diseases", "description", "symptoms"])
    writer.writeheader()
    for entry in final_data:
        writer.writerow(entry)

print(f"\n Final save in boli_nhs.json and boli_nhs.csv — total: {len(final_data)} diseases")
