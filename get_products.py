import requests
from bs4 import BeautifulSoup

def get_product_links(base_url, total_pages):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    product_links = []

    for page in range(1, total_pages + 1):
        url = f"{base_url}?page={page}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Adjust the selector based on the website's HTML structure
            links = soup.select('a.product-card__image')  # Example selector
            for link in links:
                href = link.get('href')
                product_links.append(href)
        else:
            print(f"Failed to retrieve page {page}")

    return product_links

base_url = "https://tershine.com/sv/categories/produkter"
total_pages = 5
links = get_product_links(base_url, total_pages)

# Open a file named 'product_links.txt' in write mode
with open('product_links.txt', 'w') as file:
    # Iterate through the list of links
    for link in links:
        print(link)
        # Write each link followed by a newline character
        file.write(link + '\n')