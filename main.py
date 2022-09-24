import requests
import pandas as pd
from constants import *
import numpy as np
import time
import logging

logging.basicConfig(filename="log.txt", level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

OUTPUT_NAME = "output_it.csv"

# write column names to csv
init_df = pd.DataFrame(columns=COLUMNS_OF_INTEREST)
init_df.to_csv(OUTPUT_NAME, index=False)


def scrape(scrape_page, scrape_params):
    scrape_params[PARAM_PAGE] = str(scrape_page)
    while True:
        x = requests.get(URL, params=scrape_params, headers=HEADERS)
        if x.status_code != 200:
            logging.error(f"Request failed, page: {scrape_page}")
            time.sleep(5)
            continue
        break
    json_data = x.json()

    logging.info(f"Scraping items {scrape_page * 25}/{json_data['explore_vintage']['records_matched']}")
    json_matches = json_data['explore_vintage']['matches']

    if len(json_matches) == 0:
        return False
    df = pd.json_normalize(json_matches)

    # fill in missing columns to prevent errors
    missing_columns = set(COLUMNS_OF_INTEREST) - set(df.columns.array)
    for missing_column in missing_columns:
        df[missing_column] = np.nan

    df = df[COLUMNS_OF_INTEREST]
    df.to_csv(OUTPUT_NAME, index=False, header=None, mode="a")

    if scrape_page > 80:
        logging.error(f"Params: {str(scrape_params)} went over page 80")
        return False
    return True


for i in range(1, 500):
    params = PARAMS.copy()
    params[PARAM_COUNTRY_CODE] = "IT"
    params[PARAM_PRICE_MIN] = str(i)
    params[PARAM_PRICE_MAX] = str(i + 1)

    req = requests.get(URL, params=params, headers=HEADERS)
    if req.status_code != 200 or req.json()['explore_vintage']['records_matched'] > 2000:
        for wine_type in VALUE_WINE_TYPES:
            page = 1
            params[PARAM_WINE_TYPE] = wine_type
            logging.info(f"Price {i}-{i + 1}, Wine type {wine_type}")
            while scrape(page, params):
                page += 1
    else:
        page = 1
        logging.info(f"Price {i}-{i + 1}, Wine type ALL")
        while scrape(page, params):
            page += 1
