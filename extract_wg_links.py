import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL of the page
base_url = "https://tershine.com/sv/page/tvattguider"

# Send a GET request to the page
response = requests.get(base_url)

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all elements with the class 'itc-columns__column__content'
content_divs = soup.find_all("div", class_="itc-columns__column__content")

# Use a set to collect unique links
links = set()
for div in content_divs:
    for a_tag in div.find_all("a", href=True):
        full_link = urljoin(base_url, a_tag['href'])
        links.add(full_link)  # Adding to a set automatically avoids duplicates

# Convert the set to a list for saving
links_to_crawl = list(links)

# Save links to a text file
with open("washing_guide_links.txt", "w") as file:
    for link in links_to_crawl:
        file.write(link + "\n")