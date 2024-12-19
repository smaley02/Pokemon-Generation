# import requests  
# from bs4 import BeautifulSoup

# def getdata(url):  
#     r = requests.get(url)  
#     return r.text  
    
# htmldata = getdata("https://infinitefusiondex.com/")  
# soup = BeautifulSoup(htmldata, 'html.parser')  
# for item in soup.find_all('img'): 
#     img_data = requests.get(item['src']).content
#     img_name = item['src'][54:]
#     print(img_name)
#     if img_name == '':
#         continue
#     with open(img_name, 'wb') as handler:
        # handler.write(img_data)
# collected 1025 images
        

import requests
from bs4 import BeautifulSoup
import os

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to download image
def download_image(url, directory):
    img_name = url.split('/')[-1]
    if img_name[0].isnumeric():
        print(img_name)
        with open(os.path.join(directory, img_name), 'wb') as img_file:
            img_file.write(requests.get(url).content)

# URL of the webpage containing the images
url = 'https://infinitefusiondex.com/'

# Fetch the HTML content of the webpage
response = requests.get(url)
html_content = response.content

# Parse the HTML
soup = BeautifulSoup(html_content, 'html.parser')

# Create a directory to save images
image_directory = 'images'
create_directory(image_directory)

# Find all img tags
img_tags = soup.find_all('img')

# Download each image
for img_tag in img_tags:
    img_url = img_tag.get('src')
    if img_url:
        if not img_url.startswith('http'):
            # Handle relative URLs
            img_url = url + img_url
        download_image(img_url, image_directory)

print("Images downloaded successfully!")
