import requests
import json
import pandas as pd
import urllib2
from bs4 import BeautifulSoup as BS

def get_stats(ticker):

    other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
    summary_json_response = requests.get(other_details_json_link)

    return  json.loads(summary_json_response.text)['quoteSummary']['result']


def get_prices(ticker):

    page = urllib2.urlopen('https://finance.yahoo.com/quote/{}/history?p={}'.format(ticker,ticker))
    soup = BS(page, 'html.parser')
    table = soup.find_all('table')[1]


    table_body = table.find('tbody')

    rows = table_body.find_all('tr')

    data=[]
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele]) # Get rid of empty values

    dates = [x[0] for x in data]
    close = [x[-2] for x in data]

    return zip(dates,close)