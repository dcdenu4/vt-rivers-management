import urllib
import re
import os

url = 'https://products.coastalscience.noaa.gov/habs_explorer/index.php?path=djFYWEE0NURTMllpTkN5VUVzdmtIcW5YWE83TTdDRzg2VldJWTFoc3JQdnBpckZ1K2FyTHgzUjUxSFNlOWlVZw=='
response = urllib.urlopen(url)
data = response.read()      # a `bytes` object
#print(data)

# regular expression match:
pattern = re.compile("<a href='(https.*?)'.*?a> (.*?)<")
matched_links = pattern.findall(data)
#matched_links = re.search("(\"https.*\")", data)
#print(matched_links.group(0))
print(matched_links[7])

save_dir = os.path.join("C:/\\", "Users", "ddenu", "Desktop")

# 
for i in range(8, 15):
    urllib.urlretrieve(matched_links[i][0], os.path.join(save_dir, matched_links[i][1]))