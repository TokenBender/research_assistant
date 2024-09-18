import re

def extract_arxiv_links(input_file, output_file):
    arxiv_pattern = r'https?://arxiv\.org/abs/\d+\.\d+'
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            matches = re.findall(arxiv_pattern, line)
            for match in matches:
                outfile.write(match + '\n')

# Usage
input_file = 'paper_list.txt'
output_file = 'arxiv_links.txt'
extract_arxiv_links(input_file, output_file)

print(f"Extraction complete. ArXiv links have been saved to {output_file}")