import os

# Read arxiv links from the file
with open('arxiv_links.txt', 'r') as file:
    arxiv_links = file.readlines()

# Extract arxiv IDs from the links
arxiv_ids = [link.strip().split('/')[-1] for link in arxiv_links]

# Get list of analyzed papers from the folder
analyzed_papers = os.listdir('papers_analysis_flash')
analyzed_ids = [paper.replace('.md', '') for paper in analyzed_papers]

# Find papers that are in arxiv_links but not in papers_analysis_flash
missing_papers = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in analyzed_ids]

# Print the missing papers
print("Papers present in arxiv_links but not in papers_analysis_flash:")
for paper in missing_papers:
    print(paper)
