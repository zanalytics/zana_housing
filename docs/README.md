zana_housing
==============================

Price Paid Data - Analysis

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



## Explanations of column headers in the PPD

The data is published in columns in the order set out in the table, we do not supply column headers in the files.

| Data item                         | Explanation (where appropriate)                              |
| :-------------------------------- | :----------------------------------------------------------- |
| Transaction unique identifier     | A reference number which is generated automatically recording each published sale. The number is unique and will change each time a sale is recorded. |
| Price                             | Sale price stated on the transfer deed.                      |
| Date of Transfer                  | Date when the sale was completed, as stated on the transfer deed. |
| Postcode                          | This is the postcode used at the time of the original transaction. Note that postcodes can be reallocated and these changes are not reflected in the Price Paid Dataset. |
| Property Type                     | D = Detached, S = Semi-Detached, T = Terraced, F = Flats/Maisonettes, O = Other  Note that:  - we only record the above categories to describe property type, we do not separately identify bungalows.  - end-of-terrace properties are included in the Terraced category above.  - ‘Other’ is only valid where the transaction relates to a property type that is not covered by existing values. |
| Old/New                           | Indicates the age of the property and applies to all price paid transactions, residential and non-residential. Y = a newly built property, N = an established residential building |
| Duration                          | Relates to the tenure: F = Freehold, L= Leasehold etc. Note that HM Land Registry does not record leases of 7 years or less in the Price Paid Dataset. |
| PAON                              | Primary Addressable Object Name. Typically the house number or name. |
| SAON                              | Secondary Addressable Object Name. Where a property has been divided into separate units (for example, flats), the PAON (above) will identify the building and a SAON will be specified that identifies the separate unit/flat. |
| Street                            |                                                              |
| Locality                          |                                                              |
| Town/City                         |                                                              |
| District                          |                                                              |
| County                            |                                                              |
| PPDCategory Type                  | Indicates the type of Price Paid transaction. A = Standard Price Paid entry, includes single residential property sold for value. B = Additional Price Paid entry including transfers under a power of sale/repossessions, buy-to-lets (where they can be identified by a Mortgage) and transfers to non-private individuals.  Note that category B does not separately identify the transaction types stated. HM Land Registry has been collecting information on Category A transactions from January 1995. Category B transactions were identified from October 2013. |
| Record Status - monthly file only | Indicates additions, changes and deletions to the records.(see guide below). A = Addition C = Change D = Delete.  Note that where a transaction changes category type due to misallocation (as above) it will be deleted from the original category type and added to the correct category with a new transaction unique identifier.`` |


### Record Status guide csv files

The guide highlights when additions, changes or deletions have been made to a record, this applies to the monthly file only:

- A - Added records: records added into the price paid dataset in the monthly refresh due to new sales transactions
- C - Changed records: records changed in the price paid dataset in the monthly refresh. You should replace or update records in any stored data using the unique identifier to recognise them
- D - Deleted records: records deleted from the price paid dataset in the monthly refresh. You should delete records from any stored data using the unique identifier to recognise them.

When a transaction changes category type due to misallocation it will be deleted from the original category type and added to the correct category with a new transaction unique identifier.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
