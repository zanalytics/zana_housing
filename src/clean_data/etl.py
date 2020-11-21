import requests


def raw_data(url, destination):
    r = requests.get(url, stream=True)

    with open(destination, "wb") as csv:
        for chunk in r.iter_content(chunk_size=10 ** 6):
            if chunk:
                csv.write(chunk)


# Read and write the price paid data
price_paid_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv"
price_paid_dest = "./data/raw/pp-complete.csv"
raw_data(price_paid_url, price_paid_dest)


# Read and write the HPI data
hpi_url = '''http://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/Average-prices-
                    Property-Type-2020-08.csv?utm_medium=GOV.UK&utm_source=datadownload&utm_campaign=
                    average_price_property_price&utm_term=9.30_21_10_20'''
hpi_dest = "./data/raw/house_price_index.csv"

raw_data(hpi_url, hpi_dest)