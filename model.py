from tqdm import tqdm

from get_data import get_stats, get_prices

aex_tickers=[
'AGN.AS',
'AD.AS	',
'BOKA.AS',
'AALB.AS',
'NN.AS',
'GTO.AS',
'RDSA.AS',
'VPK.AS',
'UNA.AS',
'REN.AS',
'KPN.AS',
'ASML.AS',
'INGA.AS',
'AKZA.AS',
'RAND.AS',
'DSM.AS',
'ABN.AS',
'HEIA.AS',
'SBMO.AS',
'ATC.AS',
'MT.AS',
'GLPG.AS',
'PHIA.AS',
'UL.AS',
'WKL.AS'
]

stats_list = []
prices_list = []
for ticker in tqdm(aex_tickers):
    try:
        stats = get_stats(ticker)
        prices = get_prices(ticker)

        stats_list.append(stats)
        prices_list.append(prices)
    except:
        continue

def get_features(stats_dict):

    for key, value in stats_dict.items():

        if not isinstance(value, dict):
            if isinstance(value, list):
                for nested_value in value:
                    yield get_features(nested_value)
            elif isinstance(value, dict):
                for nested_value in value:
                    yield get_features(nested_value)
            else:
                yield value
        elif isinstance(value, list):
            for nested_value in value:
                for nested_nested_value in get_features(nested_value):
                    yield nested_nested_value
        else:
            for nested_value in get_features(value):
                    yield nested_value

temp=stats_list[0][0]

features=list(get_features(temp))

