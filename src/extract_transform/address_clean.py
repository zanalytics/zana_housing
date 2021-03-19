import pandas as pd
import requests
import os






# column_names = ['id', 'price', 'date', 'postcode',
#                 'type', 'new_build', 'land', 'primary_address',
#                 'secondary_address', 'street', 'locality', 'town_city',
#                 'district', 'county', 'ppd', 'record']

# df_sample = pd.read_csv("./data/processed/pp_sample.csv", names=column_names,
#                             parse_dates=['date'])

# df_sample['primary_address'] = df_sample['primary_address'].str.title().str.replace(' - ', '-', regex=False)
# df_sample['secondary_address'] = df_sample['secondary_address'].str.title()
# df_sample['street'] = df_sample['street'].str.title()

# df_sample.loc[df_sample['secondary_address'].isnull(),'address'] = df_sample['primary_address']+', '+df_sample['street']
# df_sample.loc[df_sample['secondary_address'].notnull(), 'address'] = df_sample['secondary_address']+' '+ df_sample['primary_address']+', '+df_sample['street']

# payload={}
# headers = {
#     'Accept': 'application/json',
#     'Authorization': 'Basic Y2hyaXNwZW5AbGl2ZS5jby51azo0NGYxMzU2Mjg4NzA2ZTY0NjdkMDg2N2Y0MzBmOGM2NDNhZjc5M2Q3'
# }

# for i in range(0, len(df_sample)):
#     postcode = df_sample['postcode'][i]
#     url = f"https://epc.opendatacommunities.org/api/v1/domestic/search?postcode={postcode}"
#     response = requests.request("GET", url, headers=headers, data=payload)
#     json_response = response.json()
#     df_data = pd.DataFrame.from_dict(json_response["rows"])
#     # new data frame with split value columns 
#     new = df_data["address"].str.split(",", n = 1, expand = True) 
#     # # making separate first name column from new data frame 
#     df_data["primary_address"]= new[0] 
#     df_sample = df_sample.merge(df_data, how='left', left_on=['primary_address', 'postcode'], right_on=['primary_address', 'postcode'])
#     print(f"Progress: {i}")

# df_sample.to_csv("./data/processed/pp_epc_sample.csv", index=False)