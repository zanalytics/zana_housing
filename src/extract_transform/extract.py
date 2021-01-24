import requests
from extract_functions import raw_data, download_zip

# Extract the price paid data
price_paid_url = '''http://prod.publicdata.landregistry.gov.uk.s3-website-eu-
                    west-1.amazonaws.com/pp-complete.csv'''
price_paid_dest = "./data/raw/"
pp_file_name = "pp-complete.csv"
raw_data(price_paid_url, price_paid_dest, pp_file_name)

# Extract the HPI data
hpi_url = '''http://publicdata.landregistry.gov.uk/market-trend-data/house-
                    price-index-data/Average-prices-Property-Type-2020-08.csv?
                    utm_medium=GOV.UK&utm_source=datadownload&utm_campaign=
                    average_price_property_price&utm_term=9.30_21_10_20'''
hpi_dest = "./data/raw/"
hpi_file_name = "house_price_index.csv"
raw_data(hpi_url, hpi_dest, hpi_file_name)

# Extract the Postcodes data
download_zip("https://www.doogal.co.uk/files/postcodes.zip", "./data/raw/")
