import requests
import pandas as pd

url = "https://www.newadvent.org/bible/gen001.htm"

dfs = pd.read_html(url)

print(dfs)