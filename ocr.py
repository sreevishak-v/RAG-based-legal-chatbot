import os
import pytesseract
from PIL import Image
import cv2
import json
import re
from pdf2image import convert_from_path
from pathlib import Path
import logging
import spacy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Folder paths
INPUT_FOLDER = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/Dataset'
OUTPUT_FOLDER = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/processed_data1'
TEMP_IMAGE_FOLDER = r'C:/Users/sreevishak/Desktop/DUK/legalchatbot/temp_images'

# Create output and temp folders if they don’t exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)

# Function to convert PDF to images
def pdf_to_images(pdf_path, output_folder):
    poppler_path = r'C:/poppler/Library/bin'  # Adjust this to your Poppler bin folder
    logger.info(f"Converting PDF: {pdf_path}")
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path, dpi=300)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.png")
            image.save(image_path, 'PNG')
            image_paths.append(image_path)
            logger.info(f"Saved image: {image_path}")
        return image_paths
    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        raise

# Function to preprocess image for better OCR accuracy
def preprocess_image(image_path):
    logger.info(f"Preprocessing image: {image_path}")
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        denoised = cv2.fastNlMeansDenoising(thresh, h=30)
        temp_path = os.path.join(TEMP_IMAGE_FOLDER, f"preprocessed_{os.path.basename(image_path)}")
        cv2.imwrite(temp_path, denoised)
        logger.info(f"Wrote preprocessed image: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

# Function to extract text from an image using OCR
def ocr_extract_text(image_path):
    logger.info(f"Extracting text from: {image_path}")
    try:
        preprocessed_path = preprocess_image(image_path)
        img = Image.open(preprocessed_path)
        text = pytesseract.image_to_string(img, config='--psm 6')
        os.remove(preprocessed_path)
        logger.info(f"Removed preprocessed image: {preprocessed_path}")
        return text
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise

# Updated function to parse text into structured data
def parse_judgment_text(text):
    data = {
        "case_id": "",
        "court": "",
        "date": "",
        "judge": "",
        "petitioners": [],
        "respondents": [],
        "sections": [],
        "outcome": "",
        "full_text": text
    }

    # Case ID: Matches "Crl.MC.No. 6 of 2014 ()"
    case_id_match = re.search(r'(Crl\.MC\.No\.\s*\d+\s*of\s*\d+\s*(?:\([^)]*\))?)', text, re.IGNORECASE)
    data["case_id"] = case_id_match.group(1).strip() if case_id_match else ""

    # Court: Look for "High Court" or similar
    court_match = re.search(r'IN THE (HIGH COURT OF [A-Z\s]+(?:AT [A-Z\s]+)?)', text, re.IGNORECASE)
    data["court"] = court_match.group(1).strip() if court_match else ""

    # Date: Matches "Dated this the 2nd day of January, 2014"
    date_match = re.search(
        r'Dated\s+this\s+the\s+(\d{1,2}(?:ST|ND|RD|TH)?\s+DAY\s+OF\s+[A-Z]+\s*,\s*\d{4})', text, re.IGNORECASE)
    data["date"] = date_match.group(1).strip() if date_match else ""

    # Judge: Matches "Sd/- HARUN-UL-RASHID, JUDGE"
    judge_match = re.search(r'(?:Sd/-)?\s*([A-Z][A-Z\s.-]+),\s*JUDGE', text, re.IGNORECASE)
    data["judge"] = judge_match.group(1).strip() if judge_match else ""

    # Petitioners
    pet_match = re.search(r'PETITIONER(?:\(S\))?[/:]?\s*-?\s*([\s\S]+?)(?:\n\n|\n(?:RESPONDENTS|ORDER|BY ADV))', text, re.DOTALL)
    if pet_match:
        pet_text = pet_match.group(1).strip()
        petitioners = [line.strip() for line in pet_text.split('\n') if line.strip() and not re.match(r'BY ADV|SRI\.|SMT\.', line.strip(), re.IGNORECASE)]
        data["petitioners"] = petitioners if petitioners else [pet_text]

    # Respondents
    resp_match = re.search(
        r'RESPONDENT(?:\(S\))?[/:]?\s*-?\s*([\s\S]+?)(?:\n\n|\n(?:ORDER|THIS|BY\s+PUBLIC\s+PROSECUTOR|BY))', 
        text, re.DOTALL | re.IGNORECASE)
    if resp_match:
        resp_text = resp_match.group(1).strip()
        respondents = [line.strip() for line in resp_text.split('\n') if line.strip() and not re.match(r'BY ADV|SRI\.|SMT\.', line.strip(), re.IGNORECASE)]
        data["respondents"] = respondents if respondents else [resp_text]

    # Sections: Handle multiple formats
    sections_pattern = r'(?:under\s+)?[Ss]ection(?:s)?\s+([\d,\s\w()/]+(?:\s*(?:r/w|read\s+with)\s*[\d,\s\w()]+)?(?:\s*(?:of\s+)?(?:IPC|CrPC|CPC))?)(?:[,\s]*and\s+[\d,\s\w()/]+(?:\s*(?:r/w|read\s+with)\s*[\d,\s\w()]+)?(?:\s*(?:of\s+)?(?:IPC|CrPC|CPC))?)*'
    sec_matches = re.finditer(sections_pattern, text, re.IGNORECASE)
    sections = []
    for match in sec_matches:
        sec_text = match.group(1).strip()
        sec_parts = re.split(r',|\s+and\s+', sec_text)
        for part in sec_parts:
            part = part.strip()
            if part:
                if 'r/w' in part.lower() or 'read with' in part.lower():
                    sections.append(part)
                else:
                    sections.extend([s.strip() for s in re.split(r'\s+', part) if s.strip() and re.match(r'^\d+', s)])
    data["sections"] = list(set(sections))

    # Outcome: Target short conclusive statements after "In the result," or similar
    outcome_pattern = r'(?:In\s+the\s+result,|ORDER|For\s+the\s+reasons|Accordingly,|Hence,)\s*([\s\S]+?(?:accused\s+(?:are|is)\s+(?:discharged|acquitted|convicted)|is\s+hereby\s+(?:quashed|allowed|dismissed|disposed)|petition\s+(?:allowed|dismissed|disposed)|proceedings\s+(?:quashed|terminated)|direction\s+to\s+[\s\S]+?|bail\s+(?:granted|rejected)|No\s+interference|upheld))'
    out_match = re.search(outcome_pattern, text, re.IGNORECASE)
    if out_match:
        # Capture the outcome statement and limit to 200 characters after the match for brevity
        start_idx = out_match.start(1)
        end_idx = min(start_idx + 200, len(text))
        data["outcome"] = text[start_idx:end_idx].strip()
    else:
        # Fallback: Look for any short statement after "ORDER" or "In the result,"
        fallback_pattern = r'(?:ORDER|In\s+the\s+result,)[\s\S]+?([^\n]{1,200}(?:discharged|quashed|allowed|dismissed|disposed|granted|rejected|upheld))'
        fallback_match = re.search(fallback_pattern, text, re.IGNORECASE)
        data["outcome"] = fallback_match.group(1).strip() if fallback_match else ""

    # Enhance with NER (spaCy)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "ORG" and "court" in ent.text.lower() and not data["court"]:
            data["court"] = ent.text
        elif ent.label_ == "DATE" and not data["date"]:
            data["date"] = ent.text
        elif ent.label_ == "PERSON" and "judge" in text.lower().split(ent.text.lower())[-1] and not data["judge"]:
            data["judge"] = ent.text
        elif ent.label_ == "PERSON" and "petitioner" in text.lower().split(ent.text.lower())[-1]:
            data["petitioners"].append(ent.text)
        elif ent.label_ == "PERSON" and "respondent" in text.lower().split(ent.text.lower())[-1]:
            data["respondents"].append(ent.text)

    # Deduplicate petitioners and respondents
    data["petitioners"] = list(set(data["petitioners"]))
    data["respondents"] = list(set(data["respondents"]))

    # Log empty or problematic fields for debugging
    for key in ["case_id", "court", "date", "judge", "sections", "outcome"]:
        if not data[key]:
            logger.warning(f"Field '{key}' is empty for text starting with: {text[:50]}")

    return data

# Function to enrich existing JSON files
def enrich_existing_json(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            enriched_data = parse_judgment_text(data["full_text"])
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Enriched and saved: {output_path}")

# Main pipeline
def process_documents():
    processed_count = 0
    for filename in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, filename)
        if os.path.isfile(file_path) and filename.lower().endswith('.pdf'):
            logger.info(f"Processing: {filename}")
            image_paths = []
            try:
                image_paths = pdf_to_images(file_path, TEMP_IMAGE_FOLDER)
                full_text = ""
                for image_path in image_paths:
                    text = ocr_extract_text(image_path)
                    full_text += text + "\n"
                    os.remove(image_path)
                    logger.info(f"Removed temporary image: {image_path}")
                structured_data = parse_judgment_text(full_text)
                output_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, ensure_ascii=False, indent=4)
                logger.info(f"Saved JSON: {output_file}")
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                for img in image_paths:
                    if os.path.exists(img):
                        try:
                            os.remove(img)
                            logger.info(f"Cleaned up: {img}")
                        except Exception as cleanup_error:
                            logger.error(f"Failed to clean up {img}: {cleanup_error}")
    
    logger.info(f"Processed {processed_count} documents successfully.")
    enrich_existing_json(OUTPUT_FOLDER, OUTPUT_FOLDER)

# Run the pipeline
if __name__ == "__main__":
    process_documents()