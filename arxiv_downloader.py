import arxiv
import os
import requests
import argparse
from urllib.parse import urlparse

def download_paper_from_arxiv(url):
    """
    Download a paper from arXiv given its URL.
    
    Args:
    url (str): The arXiv URL of the paper.
    
    Returns:
    str: The path to the downloaded PDF file.
    """
    # Extract the arXiv ID from the URL
    parsed_url = urlparse(url)
    arxiv_id = parsed_url.path.split('/')[-1]
    
    # Create a client
    client = arxiv.Client()
    
    # Search for the paper using the arXiv API
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))
    
    # Create a directory to store the papers if it doesn't exist
    os.makedirs('papers', exist_ok=True)
    
    # Generate the file name
    file_name = f"papers/{paper.get_short_id()}.pdf"
    
    # Download the PDF
    paper.download_pdf(filename=file_name)
    
    print(f"Paper downloaded: {file_name}")
    return file_name

def validate_arxiv_url(url):
    """
    Validate if the given URL is a valid arXiv URL.
    
    Args:
    url (str): The URL to validate.
    
    Returns:
    bool: True if valid, False otherwise.
    """
    parsed_url = urlparse(url)
    if parsed_url.netloc in ['arxiv.org', 'www.arxiv.org']:
        if parsed_url.path.startswith('/abs/'):
            return True
        elif parsed_url.path.startswith('/pdf/'):
            return parsed_url.path.endswith('.pdf')
    return False

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download papers from arXiv.",
        epilog="Example usage: python arxiv_downloader.py https://arxiv.org/abs/2409.06927"
    )
    parser.add_argument("url", help="The arXiv URL of the paper to download.")
    return parser.parse_args()

def main():
    """
    Main function to run the CLI utility.
    """
    try:
        args = parse_arguments()
        
        if not validate_arxiv_url(args.url):
            print("Error: Invalid arXiv URL. Please provide a complete and valid URL.")
            print("For PDF links, make sure the URL ends with '.pdf'")
            print("Example usage:")
            print("  python arxiv_downloader.py https://arxiv.org/abs/2407.14679")
            print("  python arxiv_downloader.py https://arxiv.org/pdf/2407.14679.pdf")
            return
        
        downloaded_file = download_paper_from_arxiv(args.url)
        print(f"Downloaded file path: {downloaded_file}")
    except SystemExit:
        # This exception is raised by argparse when --help is used or when invalid arguments are provided
        pass
    except Exception as e:
        print(f"An error occurred while downloading the paper: {str(e)}")
        print("Example usage: python arxiv_downloader.py https://arxiv.org/abs/2407.14679")

if __name__ == "__main__":
    main()