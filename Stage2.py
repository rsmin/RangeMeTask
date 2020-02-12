import general_functions as gf

_PARAMS = gf.params_loader()

_NUMERIC_COLUMNS = _PARAMS['numeric_features']

_CATEGORY_COLUMNS = _PARAMS['category_features']

# Analyse dataset and find invalid values, The process will be presented in report
_CATEGORY_INVALID_VALUES = _PARAMS['invalid_values']


def clean_process(input_df):
    cat_format_df = gf.formatting_valid(input_df, _CATEGORY_INVALID_VALUES)
    num_format_df = gf.formatting_numeric(cat_format_df, _NUMERIC_COLUMNS)
    imputer_df = gf.imputering(num_format_df)
    return imputer_df


if __name__ == '__main__':
    input_df = gf.data_loader()
    cleaned_df = clean_process(input_df)
    print("data after clean process: \n",  cleaned_df.describe())



