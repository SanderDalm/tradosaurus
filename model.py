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
for ticker in aex_tickers:
    try:
        stats = get_stats(ticker)
        prices = get_prices(ticker)
        
        stats_list.append(stats)
        prices_list.append(prices)
    except:
        continue
