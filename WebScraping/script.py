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

START_INDEX = 0
MAX_CONDITIONS = 30

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
    # Încearcă să ia doar primele 2-3 paragrafe din containerul principal
    main_container = soup.find("main")
    if not main_container:
        main_container = soup.find("div", class_="nhsuk-width-container")

    if main_container:
        paragraphs = main_container.find_all("p", recursive=True)
        desc_paragraphs = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text.split()) > 4:
                desc_paragraphs.append(text)
            if len(desc_paragraphs) >= 2:
                break
        if desc_paragraphs:
            return "\n".join(desc_paragraphs).strip()

    return "N/A"


def extract_symptoms(soup, url,visited_urls=None):
    if visited_urls is None:
        visited_urls = set()

    if url in visited_urls:
        print(f"🔁 Already visited {url}, avoiding recursion.")
        return "N/A"
    visited_urls.add(url)

    potential_symptom_url = url.rstrip("/") + "/symptoms/"
    try:
        response = requests.get(potential_symptom_url, headers=HEADERS)
        if response.status_code == 200:
            symptoms_soup = BeautifulSoup(response.text, "html.parser")
            return extract_symptoms(symptoms_soup, potential_symptom_url, visited_urls)
    except Exception as e:
        print(f"→ Direct symptoms page not reachable: {e}")

    content = []

    current_path = url.replace(BASE_URL + "/conditions/", "").strip("/").split("/")[0]

    # Caută linkuri de tip symptoms relevante pentru boala curentă
    symptoms_link = None
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        full_link = BASE_URL + href if href.startswith("/conditions/") else href

        if (
            "symptom" in href and
            href.startswith("/conditions/") and
            current_path in href
        ):
            symptoms_link = full_link
            print(" → Found symptom link:", symptoms_link)
            break

    # Dacă există o pagină de simptome, mergem acolo și rulăm din nou
    if symptoms_link:
        try:
            time.sleep(random.uniform(1.5, 3.5))
            response = requests.get(symptoms_link, headers=HEADERS)
            if response.status_code == 200:
                symptoms_soup = BeautifulSoup(response.text, "html.parser")
                return extract_symptoms(symptoms_soup, symptoms_link, visited_urls)
            else:
                print(f" → Symptoms page not reachable: {symptoms_link}")
        except Exception as e:
            print(f" → Failed to get symptoms page: {e}")

    # Dacă nu există pagină separată, încearcă să extragi din actuala pagină
    headings = soup.find_all(["h2", "h3"])
    for heading in headings:
        heading_text = heading.get_text(strip=True).lower()
        if "symptom" in heading_text or "sign" in heading_text:
                for sibling in heading.find_next_siblings():
                        # continuă până la un heading de alt tip (ex: causes, treatment etc.)
                        if sibling.name in ["h2", "h3"] and not ("symptom" in sibling.get_text(strip=True).lower() or "sign" in sibling.get_text(strip=True).lower()):
                            break
 

                        if sibling.name == "p":
                            content.append(sibling.get_text(strip=True))
                        elif sibling.name == "ul":
                            for li in sibling.find_all("li"):
                                content.append("- " + li.get_text(strip=True))
                if content:
                    return "\n".join(content).strip()
            

    # Caz fallback - paragrafe descriptive cu fraze tipice
    paragraphs = soup.find_all("p")
    for p in paragraphs:
        p_text = p.get_text(strip=True).lower()
        
        # Dacă un paragraf conține "symptom" și e urmat de listă
        if "symptom" in p_text:
            next_tag = p.find_next_sibling()
            if next_tag and next_tag.name == "ul":
                for li in next_tag.find_all("li"):
                    content.append("- " + li.get_text(strip=True))
                return "\n".join(content).strip()


    # Ultima variantă de fallback — orice listă relevantă
    for li in soup.find_all("li"):
        li_text = li.get_text(strip=True).lower()
        if any(word in li_text for word in ["pain", "bleeding", "lump", "itch", "discharge", "swelling"]):
            content.append("- " + li.get_text(strip=True))

    return "\n".join(content).strip() if content else "N/A"

   



def scrape_condition(url):
    print(f" Scraping: {url}")
    try:
        time.sleep(random.uniform(5.5, 10.5))
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f" Page not found: {url}")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("h1").text.strip()
    except Exception as e:
        print(f" Error at title: {e}")
        return None

    return {
        "disease": title,
        "description": extract_description(soup),
        "symptoms": extract_symptoms(soup, url, set())
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

# links_to_scrape = links[START_INDEX:]

# for idx, link in enumerate(links_to_scrape, start=START_INDEX + 1):
#     try:
#         data = scrape_condition(link)
#         if data:
#             all_data.append(data)
#         print(f" {idx}/{len(links)} complete.")
#         time.sleep(random.uniform(2, 8))
#     except Exception as e:
#         print(f" Erorr at {link}: {e}")

# Test pe o singură pagină
test_url = "https://www.nhs.uk/conditions/dementia/"
data = scrape_condition(test_url)
all_data = []

if data:
    all_data.append(data)
    print("✅ Scraping test complet!")

print(f"\nDiseases length: {len(all_data)}")





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
    writer = csv.DictWriter(f, fieldnames=["disease", "description", "symptoms"])
    writer.writeheader()
    for entry in final_data:
        writer.writerow(entry)

print(f"\n Final save in boli_nhs.json and boli_nhs.csv — total: {len(final_data)} diseases")
