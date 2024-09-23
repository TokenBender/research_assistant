"""
Install the Google AI Python SDK

$ pip install google-generativeai
$ pip install psutil
$ pip install memory-profiler
"""

import os
import time
import argparse
import google.generativeai as genai
from dotenv import load_dotenv
from arxiv_downloader import download_paper_from_arxiv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions
import concurrent.futures
import threading
import logging
import tracemalloc
from memory_profiler import profile
import gc  # Imported garbage collection module
import sys  # Imported sys for exception handling
import psutil  # Imported psutil for monitoring memory usage

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("reference_main.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# Get all available API keys
API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}") 
    for i in range(1, 7) 
    if os.getenv(f"GEMINI_API_KEY_{i}")
]

if not API_KEYS:
    logging.error("No GEMINI_API_KEYs found in environment variables.")
    exit(1)

current_key_index = 0
key_lock = threading.Lock()

def rotate_api_key():
    global current_key_index
    with key_lock:
        current_key_index = (current_key_index + 1) % len(API_KEYS)
        new_key = API_KEYS[current_key_index]
        genai.configure(api_key=new_key)
    logging.info(f"Rotated to API key {current_key_index + 1}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_with_retry(arxiv_url):
    """Download paper with retry logic."""
    logging.debug(f"Attempting to download paper from URL: {arxiv_url}")
    return download_paper_from_arxiv(arxiv_url)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_to_gemini_with_retry(path, mime_type=None):
    """Uploads the given file to Gemini with retry logic."""
    logging.debug(f"Uploading file to Gemini: {path}")
    return genai.upload_file(path, mime_type=mime_type)

def should_rotate_key(exception):
    """Determine if we should rotate the API key based on the exception."""
    if isinstance(exception, requests.exceptions.HTTPError):
        return exception.response.status_code == 429
    if isinstance(exception, google_exceptions.ResourceExhausted):
        return True
    return False

@retry(
    stop=stop_after_attempt(len(API_KEYS)), 
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((google_exceptions.GoogleAPIError, requests.exceptions.RequestException))
)
def send_message_with_retry(chat_session, message):
    """Send a message to the model with retry logic and key rotation for specific errors."""
    try:
        logging.debug("Sending message to chat session.")
        return chat_session.send_message(message)
    except (google_exceptions.GoogleAPIError, requests.exceptions.RequestException) as e:
        if should_rotate_key(e):
            logging.warning(f"Resource exhausted or rate limit reached. Rotating API key. Error: {str(e)}")
            rotate_api_key()
        else:
            logging.error(f"Error encountered: {str(e)}")
        raise

def paper_already_analyzed(arxiv_id):
    """Check if the paper analysis already exists."""
    analysis_file = os.path.join("papers_analysis_flash", f"{arxiv_id}.md")
    exists = os.path.exists(analysis_file)
    logging.debug(f"Checking if paper {arxiv_id} already analyzed: {exists}")
    return exists

def paper_already_downloaded(arxiv_id):
    """Check if the paper PDF is already downloaded."""
    pdf_file = os.path.join("papers", f"{arxiv_id}.pdf")
    exists = os.path.exists(pdf_file)
    logging.debug(f"Checking if paper {arxiv_id} already downloaded: {exists}")
    return exists

def extract_arxiv_id(arxiv_url):
    """Extract arXiv ID from the URL."""
    arxiv_id = arxiv_url.split('/')[-1]
    logging.debug(f"Extracted arXiv ID: {arxiv_id} from URL: {arxiv_url}")
    return arxiv_id

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    logging.info("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            logging.debug(f"File {file.name} is still processing...")
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            error_msg = f"File {file.name} failed to process"
            logging.error(error_msg)
            raise Exception(error_msg)
    logging.info("All files are active and ready.")

def save_response_to_md(arxiv_id, response_text):
    """Saves the response to a markdown file in the papers_analysis_flash folder."""
    os.makedirs("papers_analysis_flash", exist_ok=True)
    output_file_path = os.path.join("papers_analysis_flash", f"{arxiv_id}.md")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(response_text)
    logging.info(f"Analysis saved to: {output_file_path}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze arXiv papers using Gemini AI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-u", "--url", help="The arXiv URL of the paper to analyze.")
    group.add_argument("-f", "--file", help="Path to a file containing arXiv URLs, one per line.")
    return parser.parse_args()

def read_urls_from_file(file_path):
    """Read arXiv URLs from a file, one per line."""
    logging.debug(f"Reading URLs from file: {file_path}")
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    logging.debug(f"Found {len(urls)} URLs in the file.")
    return urls

@profile
def analyze_paper(arxiv_url, model):
    """Analyze a single paper given its arXiv URL."""
    arxiv_id = extract_arxiv_id(arxiv_url)
    
    if paper_already_analyzed(arxiv_id):
        logging.info(f"Paper {arxiv_id} has already been analyzed. Skipping.")
        return

    input_file_path = None
    try:
        if paper_already_downloaded(arxiv_id):
            input_file_path = os.path.join("papers", f"{arxiv_id}.pdf")
            logging.info(f"Paper {arxiv_id} already downloaded. Using existing file.")
        else:
            input_file_path = download_with_retry(arxiv_url)
            logging.info(f"Downloaded paper {arxiv_id} to {input_file_path}")

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
                        "Analyse this paper as per your guidelines,"
                    ],
                },
            ]
        )

        response = send_message_with_retry(chat_session, "Now generate the result in markdown format.")
        logging.info(f"Analysis completed for paper {arxiv_id}")
        save_response_to_md(arxiv_id, response.text)
    except Exception as e:
        logging.error(f"Error analyzing paper {arxiv_id}: {str(e)}")
    finally:
        if input_file_path and os.path.exists(input_file_path) and not paper_already_downloaded(arxiv_id):
            os.remove(input_file_path)
            logging.debug(f"Removed temporary file: {input_file_path}")

def process_papers(urls, model, executor):
    """Process multiple papers concurrently."""
    for url in urls:
        future = executor.submit(analyze_paper, url, model)
        try:
            future.result()
            # Check memory usage after each paper
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # in MB
            logging.debug(f"Current memory usage: {memory_usage:.2f} MB")
            if memory_usage > 95:
                gc.collect()
                logging.info(f"Garbage collection invoked after memory usage exceeded 95 MB. Current usage: {memory_usage:.2f} MB")
        except Exception as exc:
            logging.error(f"{url} generated an exception: {exc}")

def log_memory_consumers(exc_type, exc_value, tb):
    """Global exception handler to log top 4 memory consumers on a memory crash."""
    if exc_type == MemoryError:
        logging.error("MemoryError encountered. Logging top 4 memory consumers.")
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            logging.debug("[ Top 4 Memory Consumers ]")
            for stat in top_stats[:4]:
                logging.debug(stat)
        except RuntimeError as e:
            logging.error(f"Failed to take tracemalloc snapshot: {e}")
    else:
        # For other exceptions, you can optionally log memory as well
        logging.error(f"Unhandled exception: {exc_type.__name__}: {exc_value}")
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            logging.debug("[ Top 4 Memory Consumers ]")
            for stat in top_stats[:4]:
                logging.debug(stat)
        except RuntimeError as e:
            logging.error(f"Failed to take tracemalloc snapshot: {e}")
    # Call the default excepthook
    sys.__excepthook__(exc_type, exc_value, tb)

def setup_exception_hook():
    """Set up a global exception hook to catch MemoryErrors and log memory usage."""
    sys.excepthook = log_memory_consumers

@profile
def main():
    tracemalloc.start()
    setup_exception_hook()  # Set up the exception hook early

    args = parse_arguments()
    
    # Initialize with the first API key
    genai.configure(api_key=API_KEYS[0])
    logging.info(f"Initialized with API key 1 of {len(API_KEYS)}")

    # Create the model
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 32768,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-exp-0827",
        generation_config=generation_config,
        system_instruction=(
            "You are an advanced AI assistant specialized in analyzing academic papers, particularly those related to deep learning and large language models (LLMs). "
            "Your task is to provide a comprehensive analysis of the given paper, with a strong emphasis on extracting and presenting the citations used by the authors "
            "to support their claims and findings. This will help in understanding the factual basis of the paper and its place within the broader research context. "
            "Please structure your analysis as follows:\n\n"
            "1. Introduction:\n"
            "   - Provide a brief overview of the paper, including its title, authors, and publication date.\n"
            "   - Summarize the main objective of the research in 1-2 sentences.\n"
            "   - Mention the total number of references cited in the paper.\n\n"
            "2. Section-by-Section Analysis with Citation Extraction:\n"
            "   - Break down the paper into its main sections (e.g., Introduction, Related Work, Methodology, Results, Discussion).\n"
            "   - For each section, provide a concise summary of the key points discussed.\n"
            "   - Crucially, identify and extract the most significant citations used by the authors in each section. For each important claim or fact, present:\n"
            "     a. The claim or fact as stated in the paper\n"
            "     b. The full citation as used by the authors, including author names, year, title, and publication venue\n"
            "     c. A brief explanation of why this citation is relevant or important to the paper's argument\n\n"
            "3. Key Insights and Supporting Literature:\n"
            "   - Identify the most important insights or findings presented in the paper.\n"
            "   - For each key insight, list the primary citations the authors used to support or contextualize their findings.\n"
            "   - Explain how these cited works contribute to the paper's arguments or findings.\n\n"
            "4. Experimental Methodology and Its Foundations:\n"
            "   - Describe the experimental setup used in the paper.\n"
            "   - Identify any cited works that the authors used as a basis for their methodology.\n"
            "   - Highlight any novel aspects of the methodology and note whether the authors cite any works to justify these novel approaches.\n\n"
            "5. Results in Context:\n"
            "   - Summarize the main results of the paper.\n"
            "   - For each significant result, identify any citations the authors used to compare their findings with existing literature.\n"
            "   - Note any instances where the authors' results confirm, contradict, or extend cited works.\n\n"
            "6. Discussion and Related Work:\n"
            "   - Analyze how the authors situate their work within the existing literature.\n"
            "   - Identify the key papers cited in the discussion or related work section.\n"
            "   - Explain how the authors use these citations to highlight the novelty or importance of their own work.\n\n"
            "7. Future Work and Open Questions:\n"
            "   - Identify areas for further research suggested by the authors.\n"
            "   - Note any citations used to support these suggestions for future work.\n\n"
            "8. Critical Analysis of Citation Usage:\n"
            "   - Evaluate how effectively the authors use citations to support their arguments.\n"
            "   - Identify any areas where additional citations might have been beneficial.\n"
            "   - Note any potential biases in the selection of cited works (e.g., over-reliance on certain authors or publications).\n\n"
            "9. Final Summary:\n"
            "   - Offer a concise overview of the paper's contribution to the field.\n"
            "   - Highlight the most influential or frequently cited works used throughout the paper.\n"
            "   - Provide your assessment of how well the paper integrates existing literature to support its claims and findings.\n\n"
            "Throughout your analysis, prioritize the extraction and presentation of citations used within the paper. This approach will help readers understand the factual basis of the research, its relationship to existing literature, and the broader context of the work. Your goal is to create a comprehensive map of the cited literature that supports the paper's arguments and findings, enabling readers to trace the origins of key ideas and assess the paper's contribution to the field.\n\n"
            "Please ensure that your analysis is thorough, objective, and tailored to an audience with standard deep learning knowledge. Use technical terms appropriately, but provide brief explanations where necessary. Your analysis should serve as a guide to understanding not just the paper itself, but also the network of research upon which it builds."
        ),
    )

    if args.url:
        analyze_paper(args.url, model)
    elif args.file:
        urls = read_urls_from_file(args.file)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            process_papers(urls, model, executor)

    # Snapshot memory allocation
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    logging.debug("[ Top 4 Memory Consumers ]")
    for stat in top_stats[:4]:
        logging.debug(stat)

if __name__ == "__main__":
    main()