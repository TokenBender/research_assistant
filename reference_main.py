"""
Install the Google AI Python SDK

$ pip install google-generativeai
"""

import os
import time
import argparse
import google.generativeai as genai
from dotenv import load_dotenv
from arxiv_downloader import download_paper_from_arxiv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_with_retry(arxiv_url):
    """Download paper with retry logic."""
    return download_paper_from_arxiv(arxiv_url)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_to_gemini_with_retry(path, mime_type=None):
    """Uploads the given file to Gemini with retry logic."""
    return genai.upload_file(path, mime_type=mime_type)

def paper_already_analyzed(arxiv_id):
    """Check if the paper analysis already exists."""
    analysis_file = os.path.join("paper_analysis", f"{arxiv_id}.md")
    return os.path.exists(analysis_file)

def extract_arxiv_id(arxiv_url):
    """Extract arXiv ID from the URL."""
    return arxiv_url.split('/')[-1]

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
    print()

def save_response_to_md(arxiv_id, response_text):
    """Saves the response to a markdown file in the paper_analysis folder."""
    os.makedirs("paper_analysis", exist_ok=True)
    output_file_path = os.path.join("paper_analysis", f"{arxiv_id}.md")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(response_text)
    print(f"Analysis saved to: {output_file_path}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze arXiv papers using Gemini AI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="The arXiv URL of the paper to analyze.")
    group.add_argument("-f", "--file", help="Path to a file containing arXiv URLs, one per line.")
    return parser.parse_args()

def read_urls_from_file(file_path):
    """Read arXiv URLs from a file, one per line."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def analyze_paper(arxiv_url, model):
    """Analyze a single paper given its arXiv URL."""
    arxiv_id = extract_arxiv_id(arxiv_url)
    
    if paper_already_analyzed(arxiv_id):
        print(f"Paper {arxiv_id} has already been analyzed. Skipping.")
        return

    try:
        input_file_path = download_with_retry(arxiv_url)
    except Exception as e:
        print(f"Error downloading paper from {arxiv_url} after retries: {str(e)}")
        return

    try:
        files = [
            upload_to_gemini_with_retry(input_file_path, mime_type="application/pdf"),
        ]

        # Wait for files to be processed
        wait_for_files_active(files)

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        files[0],
                        "Analyse this paper as per your guidelines,",
                    ],
                },
            ]
        )

        response = chat_session.send_message("Now generate the result in markdown format.")

        print(response.text)

        # Save the response to a markdown file
        save_response_to_md(arxiv_id, response.text)
    except Exception as e:
        print(f"Error analyzing paper {arxiv_id}: {str(e)}")
    finally:
        # Clean up the downloaded file
        if os.path.exists(input_file_path):
            os.remove(input_file_path)

def main():
    args = parse_arguments()
    
    # Create the model
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 32768,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-exp-0827",
        generation_config=generation_config,
        system_instruction="You are an advanced AI assistant specialized in analyzing academic papers, particularly those related to deep learning and large language models (LLMs). Your task is to provide a comprehensive analysis of the given paper, with a strong emphasis on extracting and presenting the citations used by the authors to support their claims and findings. This will help in understanding the factual basis of the paper and its place within the broader research context. Please structure your analysis as follows:\n\n1. Introduction:\n   - Provide a brief overview of the paper, including its title, authors, and publication date.\n   - Summarize the main objective of the research in 1-2 sentences.\n   - Mention the total number of references cited in the paper.\n\n2. Section-by-Section Analysis with Citation Extraction:\n   - Break down the paper into its main sections (e.g., Introduction, Related Work, Methodology, Results, Discussion).\n   - For each section, provide a concise summary of the key points discussed.\n   - Crucially, identify and extract the most significant citations used by the authors in each section. For each important claim or fact, present:\n     a. The claim or fact as stated in the paper\n     b. The full citation as used by the authors, including author names, year, title, and publication venue\n     c. A brief explanation of why this citation is relevant or important to the paper's argument\n\n3. Key Insights and Supporting Literature:\n   - Identify the most important insights or findings presented in the paper.\n   - For each key insight, list the primary citations the authors used to support or contextualize their findings.\n   - Explain how these cited works contribute to the paper's arguments or findings.\n\n4. Experimental Methodology and Its Foundations:\n   - Describe the experimental setup used in the paper.\n   - Identify any cited works that the authors used as a basis for their methodology.\n   - Highlight any novel aspects of the methodology and note whether the authors cite any works to justify these novel approaches.\n\n5. Results in Context:\n   - Summarize the main results of the paper.\n   - For each significant result, identify any citations the authors used to compare their findings with existing literature.\n   - Note any instances where the authors' results confirm, contradict, or extend cited works.\n\n6. Discussion and Related Work:\n   - Analyze how the authors situate their work within the existing literature.\n   - Identify the key papers cited in the discussion or related work section.\n   - Explain how the authors use these citations to highlight the novelty or importance of their own work.\n\n7. Future Work and Open Questions:\n   - Identify areas for further research suggested by the authors.\n   - Note any citations used to support these suggestions for future work.\n\n8. Critical Analysis of Citation Usage:\n   - Evaluate how effectively the authors use citations to support their arguments.\n   - Identify any areas where additional citations might have been beneficial.\n   - Note any potential biases in the selection of cited works (e.g., over-reliance on certain authors or publications).\n\n9. Final Summary:\n   - Offer a concise overview of the paper's contribution to the field.\n   - Highlight the most influential or frequently cited works used throughout the paper.\n   - Provide your assessment of how well the paper integrates existing literature to support its claims and findings.\n\nThroughout your analysis, prioritize the extraction and presentation of citations used within the paper. This approach will help readers understand the factual basis of the research, its relationship to existing literature, and the broader context of the work. Your goal is to create a comprehensive map of the cited literature that supports the paper's arguments and findings, enabling readers to trace the origins of key ideas and assess the paper's contribution to the field.\n\nPlease ensure that your analysis is thorough, objective, and tailored to an audience with standard deep learning knowledge. Use technical terms appropriately, but provide brief explanations where necessary. Your analysis should serve as a guide to understanding not just the paper itself, but also the network of research upon which it builds.",
    )

    if args.url:
        analyze_paper(args.url, model)
    elif args.file:
        urls = read_urls_from_file(args.file)
        for url in urls:
            print(f"Analyzing paper from URL: {url}")
            analyze_paper(url, model)
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()