import os
import json
import re
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Folder paths
INPUT_FOLDER = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/processed_data1'
OUTPUT_FOLDER = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/cleaned_data'

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Key fields to preprocess
KEY_FIELDS = ["case_id", "court", "date", "judge", "petitioners", "respondents", "sections", "outcome"]

def clean_text(text):
    """Remove OCR artifacts and extra whitespace."""
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII chars
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def preprocess_json(data):
    full_text = clean_text(data.get("full_text", ""))
    cleaned_data = data.copy()

    # 1. case_id
    if not cleaned_data["case_id"]:
        case_id_match = re.search(r'(Crl\.MC\.No\.|CRL\.MC\s+NO\.|W\.P\.|SC|CC)\s*[\d/]+\s*(?:of\s*\d+)?\s*(?:\([^)]*\))?', full_text, re.IGNORECASE)
        cleaned_data["case_id"] = case_id_match.group(0).strip() if case_id_match else ""

    # 2. court
    court_match = re.search(r'(HIGH COURT OF [A-Z\s]+(?:AT [A-Z\s]+)?)', full_text, re.IGNORECASE)
    cleaned_data["court"] = court_match.group(1).strip() if court_match else cleaned_data["court"].split('\n')[0].strip()

    # 3. date
    date_match = re.search(r'Dated\s+this\s+the\s+(\d{1,2}(?:ST|ND|RD|TH)?\s+DAY\s+OF\s+[A-Z]+\s*,\s*\d{4})', full_text, re.IGNORECASE)
    cleaned_data["date"] = date_match.group(1).strip() if date_match else cleaned_data["date"]

    # 4. judge
    judge_match = re.search(r'(?:JUSTICE|Judge)\s+([A-Z][A-Z\s.-]+)(?:,\s*JUDGE)?', full_text, re.IGNORECASE)
    if not judge_match:
        judge_match = re.search(r'Sd/-\s*([A-Z][A-Z\s.-]+)', full_text, re.IGNORECASE)
    cleaned_data["judge"] = judge_match.group(1).strip() if judge_match else cleaned_data["judge"]

    # 5. petitioners and respondents
    def clean_names(name_list):
        cleaned = []
        for name in name_list:
            name = clean_text(name)
            if not re.search(r'BY\s+ADV|SRI\.|SMT\.|PIN|PUBLIC\s+PROSECUTOR|COURT|ACCUSED|COMPLAINANT|STATE', name, re.IGNORECASE):
                # Extract name-like patterns
                name_match = re.search(r'([A-Z][A-Za-z\s.-]+)(?:,\s*AGED\s*\d+)?', name, re.IGNORECASE)
                if name_match:
                    cleaned.append(name_match.group(1).strip())
        return list(set(cleaned))  # Deduplicate

    cleaned_data["petitioners"] = clean_names(cleaned_data["petitioners"])
    cleaned_data["respondents"] = clean_names(cleaned_data["respondents"])

    # 6. sections
    sections_pattern = r'(?:under\s+)?[Ss]ection(?:s)?\s+([\d,\s\w()/]+(?:\s*(?:r/w|read\s+with)\s*[\d,\s\w()]+)?(?:\s*(?:of\s+)?(?:IPC|CrPC|CPC|I\.T\.\s*Act))?)(?:[,\s]*and\s+[\d,\s\w()/]+)?'
    sec_matches = re.finditer(sections_pattern, full_text, re.IGNORECASE)
    sections = []
    for match in sec_matches:
        sec_text = match.group(1).strip()
        sec_parts = re.split(r',|\s+and\s+', sec_text)
        for part in sec_parts:
            part = part.strip()
            if 'r/w' in part.lower() or 'read with' in part.lower():
                sections.append(part)
            else:
                sections.extend([s.strip() for s in re.split(r'\s+', part) if re.match(r'^\d+', s)])
    cleaned_data["sections"] = list(set(sections)) if sections else cleaned_data["sections"]

    # 7. outcome
    outcome_pattern = r'(?:In\s+the\s+result,|ORDER|Accordingly,)\s*([\s\S]+?(?:quashed|allowed|dismissed|disposed|discharged|granted|rejected|bail|no\s+interference))'
    out_match = re.search(outcome_pattern, full_text, re.IGNORECASE)
    cleaned_data["outcome"] = out_match.group(1).strip()[:200] if out_match else cleaned_data["outcome"]

    # 8. full_text (optional cleaning)
    cleaned_data["full_text"] = full_text

    return cleaned_data

def preprocess_files():
    json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} JSON files to preprocess.")

    for json_file in json_files:
        input_path = os.path.join(INPUT_FOLDER, json_file)
        output_path = os.path.join(OUTPUT_FOLDER, json_file)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cleaned_data = preprocess_json(data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Preprocessed and saved: {json_file}")
        
        except Exception as e:
            logger.error(f"Error preprocessing {json_file}: {e}")

if __name__ == "__main__":
    preprocess_files()