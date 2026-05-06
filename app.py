from flask import Flask, render_template, request, jsonify
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re

app = Flask(__name__)


genai.configure(api_key="AIzaSyDEFL_DhNhcA-AqeoZl6B4I3hsPnc2lpB8")


embed_model = SentenceTransformer('all-MiniLM-L6-v2')

texts = []
index = None


def extract_text(pdf):
    text = ""

    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def create_vector_db(text):
    global texts, index

    chunk_size = 300
    overlap = 50

    texts = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]
        texts.append(chunk)
        start += chunk_size - overlap

    if not texts:
        index = None
        return

    embeddings = embed_model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))


def search(query):
    if index is None:
        return ""

    query_vec = embed_model.encode([query])

    D, I = index.search(np.array(query_vec), k=5)

    return " ".join(
        [texts[i] for i in I[0] if i < len(texts)]
    )

def extract_issue_count(text, label):
    pattern = rf"{label}.*?(\d+)"
    match = re.search(
        pattern,
        text,
        re.IGNORECASE | re.DOTALL
    )

    return int(match.group(1)) if match else 0


def dast_summary(text):
    critical = extract_issue_count(
        text,
        "critical severity issues"
    )

    high = extract_issue_count(
        text,
        "high severity issues"
    )

    medium = extract_issue_count(
        text,
        "medium severity issues"
    )

    info = extract_issue_count(
        text,
        "informational severity issues"
    )

    total = extract_issue_count(
        text,
        "total security issues"
    )

    if total == 0:
        total = critical + high + medium + info

    risk_score = (
        critical * 4 +
        high * 3 +
        medium * 2 +
        info
    )

    return f"""
DAST Report Summary
----------------------
Critical : {critical}
High     : {high}
Medium   : {medium}
Info     : {info}
Total    : {total}
Risk Score : {risk_score}
"""

def extract_vulnerabilities(text):
    text = text.lower()

    vulnerability_patterns = {
        "SQL Injection": (
            "High",
            ["sql injection", "sqli"]
        ),

        "Cross-Site Scripting (XSS)": (
            "High",
            ["xss", "cross-site scripting"]
        ),

        "Command Injection": (
            "High",
            ["command injection"]
        ),

        "Broken Authentication": (
            "High",
            ["broken authentication"]
        ),

        "CSRF": (
            "Medium",
            ["csrf"]
        ),

        "Insecure Cookies": (
            "Medium",
            ["insecure cookie"]
        ),

        "Missing Security Headers": (
            "Low",
            ["security header"]
        ),

        "Open Redirect": (
            "Low",
            ["open redirect"]
        )
    }

    vulns = []

    for vuln_name, (severity, keywords) in vulnerability_patterns.items():
        count = 0

        for keyword in keywords:
            count += text.count(keyword)

        if count > 0:
            vulns.append({
                "name": vuln_name,
                "severity": severity,
                "count": count
            })

    return vulns

def get_solution_steps(vuln_name):
    solutions = {
        "SQL Injection": [
            "Use parameterized queries",
            "Validate input"
        ],

        "Cross-Site Scripting (XSS)": [
            "Sanitize inputs",
            "Escape output"
        ],

        "Command Injection": [
            "Avoid shell execution",
            "Sanitize commands"
        ],

        "Broken Authentication": [
            "Use secure sessions",
            "Enable MFA"
        ],

        "CSRF": [
            "Add CSRF tokens",
            "Validate origin"
        ],

        "Insecure Cookies": [
            "Enable Secure flag",
            "Enable HttpOnly"
        ],

        "Missing Security Headers": [
            "Add HSTS",
            "Add CSP"
        ],

        "Open Redirect": [
            "Validate redirect URLs",
            "Use whitelist"
        ]
    }

    return solutions.get(
        vuln_name,
        ["Follow OWASP guidelines"]
    )


def filter_by_severity(text, severity_level):
    vulns = extract_vulnerabilities(text)

    filtered = [
        vuln for vuln in vulns
        if vuln["severity"].lower() ==
        severity_level.lower()
    ]

    if not filtered:
        return (
            f"No {severity_level} "
            f"severity vulnerabilities found."
        )

    result = (
        f" {severity_level.upper()} "
        f"Severity Vulnerabilities:\n\n"
    )

    total_count = 0

    for vuln in filtered:
        total_count += vuln["count"]

        result += f" {vuln['name']}\n"
        result += f"   Count: {vuln['count']}\n"

        steps = get_solution_steps(vuln["name"])

        for i, step in enumerate(steps, 1):
            result += f"   Step {i}: {step}\n"

        result += "\n"

    result += (
        f"Total {severity_level} "
        f"Issues: {total_count}"
    )

    return result


def ask_gemini(context, question):
    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        prompt = f"""

        Use ONLY the context below.

        Context:
        {context}

        Question:
        {question}
        """

        response = model.generate_content(prompt)

        return (
            response.text
            if response.text
            else "No response"
        )

    except Exception as e:
        return str(e)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    text = extract_text(file)

    create_vector_db(text)

    return dast_summary(text)


@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json["message"].lower()

    if index is None:
        return jsonify({
            "reply": "Upload PDF first!"
        })

    full_text = " ".join(texts)

    if "summary" in msg:
        return jsonify({
            "reply": dast_summary(full_text)
        })

    elif "high" in msg:
        return jsonify({
            "reply": filter_by_severity(
                full_text,
                "High"
            )
        })

    elif "medium" in msg:
        return jsonify({
            "reply": filter_by_severity(
                full_text,
                "Medium"
            )
        })

    elif "low" in msg:
        return jsonify({
            "reply": filter_by_severity(
                full_text,
                "Low"
            )
        })

    else:
        context = search(msg)

        answer = ask_gemini(context, msg)

        return jsonify({
            "reply": answer
        })


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)