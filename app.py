# ----------------------
# app.py - PDF to ITP Activity Matcher (Streamlit + Google Drive)
# ----------------------

import streamlit as st
import fitz
from pdf2image import convert_from_path
from pytesseract import image_to_string
import pandas as pd
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
import os, tempfile
import nltk
from tqdm import tqdm

# Google Drive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

nltk.download('stopwords')

# ----------------------
# Initialize stemmer, stopwords, model
# ----------------------
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------
# Core activity keywords
# ----------------------
core_activity_keywords = [
    "excavation", "wiring", "termination", "installation", "testing", "grounding",
    "material", "drawing", "prequalification", "storage", "handling", "panel", "device", 
    "continuity", "insulation", "fire", "handover", "collation", "asbuilt"
]
core_activity_keywords = [stemmer.stem(w) for w in core_activity_keywords]

# ----------------------
# Helper functions
# ----------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words and not w.isdigit()]
    return tokens

def extract_itp_from_pdf(pdf_path, pages=1):
    doc = fitz.open(pdf_path)
    text = ""
    for i in range(min(pages, len(doc))):
        page_text = doc[i].get_text()
        if page_text.strip():
            text += page_text
        else:
            images = convert_from_path(pdf_path, dpi=300, first_page=i+1, last_page=i+1)
            for image in images:
                text += image_to_string(image)
    match = re.search(r'([A-Z\-]*ITP[-\s]?\d+([-]\d+)?)', text, re.IGNORECASE)
    return match.group(0).upper() if match else None

def filter_candidates_by_core_keywords(title, activities_list):
    title_tokens = preprocess_text(title)
    title_keywords = set([t for t in title_tokens if t in core_activity_keywords])
    candidates = []
    for activity in activities_list:
        activity_tokens = preprocess_text(activity)
        activity_keywords = set([t for t in activity_tokens if t in core_activity_keywords])
        if title_keywords & activity_keywords:
            candidates.append(activity)
    return candidates

def token_semantic_match(title, activities_list):
    if not activities_list:
        return "", 0, 0

    title_tokens = preprocess_text(title)
    title_emb = model.encode(title, convert_to_tensor=True)
    activity_embs = model.encode(activities_list, convert_to_tensor=True)
    cosine_scores = util.cos_sim(title_emb, activity_embs)[0].tolist()

    candidates = []
    for idx, activity in enumerate(activities_list):
        activity_tokens = preprocess_text(activity)
        common_tokens = set(title_tokens) & set(activity_tokens)
        token_score = len(common_tokens)
        candidates.append({
            "Activity": activity,
            "Token Score": token_score,
            "Semantic Score": cosine_scores[idx]*100
        })

    best_row = max(candidates, key=lambda x: (x['Token Score'], x['Semantic Score']))
    return best_row['Activity'], best_row['Token Score'], best_row['Semantic Score']

def normalize_rev(rev):
    if pd.isna(rev):
        return ""
    return re.sub(r'[\s_\-]', '', str(rev).upper())

def match_title_and_revision(pdf_name, work_df):
    base_name = os.path.splitext(os.path.basename(pdf_name))[0].upper()
    parts = base_name.split('-')
    if len(parts) >= 2:
        rev_pdf = '-'.join(parts[-2:])
        doc_no_pdf = '-'.join(parts[:-2])
    else:
        doc_no_pdf = base_name
        rev_pdf = ""

    doc_no_pdf_norm = re.sub(r'[\s_\-]', '', doc_no_pdf)
    rev_pdf_norm = normalize_rev(rev_pdf)

    for idx, row in work_df.iterrows():
        doc_no_excel_norm = re.sub(r'[\s_\-]', '', str(row['Document No']).upper())
        rev_excel_norm = normalize_rev(row['Rev']) if 'Rev' in row else ""
        if doc_no_pdf_norm == doc_no_excel_norm and rev_pdf_norm == rev_excel_norm:
            return row['Title'], row['Rev']
    return "", ""

# ----------------------
# Streamlit UI
# ----------------------
st.title("PDF to ITP Activity Matcher (Google Drive)")

work_file = st.file_uploader("Upload Work Inspection Excel (Document No, Title, Rev)", type=["xlsx"])
itp_file = st.file_uploader("Upload ITP Activities Excel (ITP Number & Activity)", type=["xlsx"])
drive_folder_id = st.text_input("Enter Google Drive Folder ID with PDFs:")

if work_file and itp_file and drive_folder_id:

    # Load Excel files
    work_df = pd.read_excel(work_file)
    work_df['Document No'] = work_df['Document No'].astype(str)
    itp_df = pd.read_excel(itp_file)
    itp_df['ITP Number'] = itp_df['ITP Number'].astype(str)

    # Google Drive Authentication
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    # List PDFs in Drive folder
    file_list = drive.ListFile({'q': f"'{drive_folder_id}' in parents and trashed=false"}).GetList()
    pdf_paths = []
    temp_dir = tempfile.mkdtemp()
    for f in file_list:
        if f['title'].lower().endswith('.pdf'):
            f_path = os.path.join(temp_dir, f['title'])
            f.GetContentFile(f_path)
            pdf_paths.append(f_path)

    st.write(f"Found {len(pdf_paths)} PDFs in Google Drive folder. Processing...")

    # ----------------------
    # Process PDFs
    # ----------------------
    final_rows = []
    matched_activities_set = set()

    progress = st.progress(0)
    for idx, pdf_file in enumerate(pdf_paths):
        extracted_itp = extract_itp_from_pdf(pdf_file)
        matched_title, matched_rev = match_title_and_revision(pdf_file, work_df)
        best_itp_number = None
        if extracted_itp:
            best_itp_number, score, _ = process.extractOne(extracted_itp, itp_df['ITP Number'].tolist(), scorer=fuzz.ratio)
            if score < 70:
                best_itp_number = None

        activities_list = itp_df[itp_df['ITP Number'] == best_itp_number]['Activity'].tolist() if best_itp_number else []
        filtered_activities = filter_candidates_by_core_keywords(matched_title, activities_list)
        matched_activity, token_score, semantic_score = token_semantic_match(matched_title, filtered_activities)

        final_rows.append({
            "Document No": os.path.splitext(os.path.basename(pdf_file))[0],
            "Title": matched_title,
            "Rev": matched_rev,
            "Extracted ITP": extracted_itp if extracted_itp else "",
            "ITP Number": best_itp_number if best_itp_number else "Not Found",
            "Matched Activity": matched_activity,
            "Token Score": token_score,
            "Semantic Score": semantic_score
        })

        if best_itp_number and matched_activity:
            matched_activities_set.add((best_itp_number, matched_activity))

        progress.progress((idx + 1) / len(pdf_paths))

    # ----------------------
    # Download buttons
    # ----------------------
    final_df = pd.DataFrame(final_rows)
    st.download_button("Download Matched Results", final_df.to_excel(index=False), file_name="final_matched_results.xlsx")

    unmatched_rows = []
    for idx, row in itp_df.iterrows():
        itp_number = row['ITP Number']
        activity = row['Activity']
        if (itp_number, activity) not in matched_activities_set:
            unmatched_rows.append({
                "ITP Number": itp_number,
                "Activity": activity
            })

    unmatched_df = pd.DataFrame(unmatched_rows)
    st.download_button("Download Unmatched Activities", unmatched_df.to_excel(index=False), file_name="unmatched_activities.xlsx")

