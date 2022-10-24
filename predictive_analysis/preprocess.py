import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('trimmed_data_2.csv')
df2 = df[['vintage.id', 'vintage.statistics.wine_ratings_average', 'vintage.wine.type_id',
'vintage.wine.taste.structure.acidity', 'vintage.wine.taste.structure.intensity',
'vintage.wine.taste.structure.sweetness', 'vintage.wine.taste.structure.tannin',
'price.amount', 'vintage.wine.has_valid_ratings', 'vintage.year',
'price.bottle_type.volume_ml', 'price_category']].copy()
df_valid = df2[df2['vintage.wine.has_valid_ratings'] == True]

df_valid = df_valid.rename(columns={'vintage.id': 'id', 'vintage.statistics.wine_ratings_average': 'Rating_avg',
'vintage.wine.type_id': 'Type_id', 'vintage.wine.taste.structure.acidity': 'Acidity',
'vintage.wine.taste.structure.intensity': 'Intensity', 'vintage.wine.taste.structure.sweetness': 'Sweetness',
'vintage.wine.taste.structure.tannin': 'Tannin', 'vintage.wine.has_valid_ratings': 'Valid_ratings',
'price.amount': 'Price', 'price.bottle_type.volume_ml': 'Volume_ml'})

# print(df.isnull().any())
# boolean = df['vintage.id'].duplicated().any()
# print(boolean)
# print(df.dtypes)
#print(df['price.bottle_type.volume_ml'].nunique())

#print(df_valid)

df_valid['vintage.year'] = df_valid['vintage.year'].astype(str)
#print(df_valid)
df_valid = df_valid.drop(df_valid[df_valid['vintage.year'] == 'N.V.'].index)
#print(df_valid)
#print(df_valid['vintage.year'].isna().sum())
df_valid['vintage.year'] = df_valid['vintage.year'].astype(float)

#df_valid = df_valid[df_valid['vintage.year'] == 'N.V.']

df_valid_uniform = df_valid[df_valid['Volume_ml'] == 750]

#df_valid_uniform = df_valid_uniform[df_valid_uniform['vintage.year'] != 'N.V.']

#print(df_valid)
#print(df_valid_uniform)

# All of these have null values in all taste structure columns
#df_valid['vintage.year'] = df_valid['vintage.year'].astype(str).astype(int)
df_red = df_valid[df_valid['Type_id'] == 1]
df_white = df_valid[df_valid['Type_id'] == 2]
df_red_uniform = df_valid_uniform[df_valid_uniform['Type_id'] == 1]
df_white_uniform = df_valid_uniform[df_valid_uniform['Type_id'] == 2]
# Tannin is not used for white wines so dropping this
df_white_uniform = df_white_uniform.drop(['Tannin'], axis=1)
df_white = df_white.drop(['Tannin'], axis=1)

# Won't be using these
# df_sparkling = df2[df2['vintage.wine.type_id'] == 3]
# df_rose = df2[df2['vintage.wine.type_id'] == 4]
# df_dessert = df2[df2['vintage.wine.type_id'] == 7]
# df_fortified = df2[df2['vintage.wine.type_id'] == 24]

# Created these to check how many rows had null values
df_red_null = df_red[df_red.isna().any(axis=1)]
df_white_null = df_white[df_white.isna().any(axis=1)]


df_red = df_red.dropna()
df_white = df_white.dropna()
df_red_uniform = df_red_uniform.dropna()
df_white_uniform = df_white_uniform.dropna()

#print(df_red)

#df_red['vintage.year'] = df_red['vintage.year'].astype(str)
#df_white['vintage.year'] = df_white['vintage.year'].astype(str)

#df_red['vintage.year'] = pd.to_numeric(df_red['vintage.year'])


#print(df_red)
#print(df_white)
#print(df_red_uniform)
#print(df_white_uniform)

#print(df_valid.price_category.unique())
#print(df_valid['price_category'].value_counts())

#asdf = df[df['vintage.id'] == 1476911]

#print(asdf)


#print(df_red['vintage.year'])
#print(df_red.dtypes)

#red_corr = df_red.corr()

#print(red_corr)

df_red_corr = df_red[['id', 'Price', 'Rating_avg', 'Acidity', 'Intensity', 'Sweetness', 'Tannin', 'vintage.year', 'price_category']].copy()
df_white_corr = df_white[['id', 'Price', 'Rating_avg', 'Acidity', 'Intensity', 'Sweetness', 'vintage.year', 'price_category']].copy()
df_red_corr_uniform = df_red_uniform[['id', 'Price', 'Rating_avg', 'Acidity', 'Intensity', 'Sweetness', 'Tannin', 'vintage.year', 'price_category']].copy()
df_white_corr_uniform = df_white_uniform[['id', 'Price', 'Rating_avg', 'Acidity', 'Intensity', 'Sweetness', 'vintage.year', 'price_category']].copy()

df_red_corr.to_csv('red_data.csv', index=False)
df_white_corr.to_csv('white_data.csv', index=False)
df_red_corr_uniform.to_csv('red_data_uniform.csv', index=False)
df_white_corr_uniform.to_csv('white_data_uniform.csv', index=False)

#df_red_corr = df_red[['vintage.statistics.wine_ratings_average', 'vintage.wine.taste.structure.acidity',
#'vintage.wine.taste.structure.intensity','vintage.wine.taste.structure.sweetness',
#'vintage.wine.taste.structure.tannin', 'price.amount', 'vintage.year']].copy()

#df_white_corr = df_white[['vintage.statistics.wine_ratings_average',
#'vintage.wine.taste.structure.acidity','vintage.wine.taste.structure.intensity',
#'vintage.wine.taste.structure.sweetness', 'price.amount', 'vintage.year']].copy()


# print(df_red.shape)
# print(df_red_null.shape)
# print(df_white.shape)
# print(df_white_null.shape)

# print(df2['vintage.wine.taste.structure.acidity'].isna().sum())        # 11241 total
# print(df2['vintage.wine.taste.structure.fizziness'].isna().sum())      # 73786 total
# print(df2['vintage.wine.taste.structure.intensity'].isna().sum())      # 11241 total
# print(df2['vintage.wine.taste.structure.sweetness'].isna().sum())      # 13100 total
# print(df2['vintage.wine.taste.structure.tannin'].isna().sum())         # 33649 total

# 45778 ROWS
# All null values for fizziness (as expected), other null values are likely from the same rows
# print(df_red['vintage.wine.taste.structure.acidity'].isna().sum())      # 3791 total
# print(df_red['vintage.wine.taste.structure.fizziness'].isna().sum())    # 45778 total
# print(df_red['vintage.wine.taste.structure.intensity'].isna().sum())    # 3791 total
# print(df_red['vintage.wine.taste.structure.sweetness'].isna().sum())    # 3791 total
# print(df_red['vintage.wine.taste.structure.tannin'].isna().sum())       # 3796 total

# 22062 ROWS
# Almost all missing tannin (looks like its only applicable for red wine)
# print(df_white['vintage.wine.taste.structure.acidity'].isna().sum())      # 2874 total
# print(df_white['vintage.wine.taste.structure.fizziness'].isna().sum())    # 22061 total
# print(df_white['vintage.wine.taste.structure.intensity'].isna().sum())    # 2874 total
# print(df_white['vintage.wine.taste.structure.sweetness'].isna().sum())    # 2875 total
# print(df_white['vintage.wine.taste.structure.tannin'].isna().sum())       # 22060 total

# 3072 ROWS
# Roughly 1/3 with null values
# print(df_sparkling['vintage.wine.taste.structure.acidity'].isna().sum())      # 1203 total
# print(df_sparkling['vintage.wine.taste.structure.fizziness'].isna().sum())    # 1214 total
# print(df_sparkling['vintage.wine.taste.structure.intensity'].isna().sum())    # 1203 total
# print(df_sparkling['vintage.wine.taste.structure.sweetness'].isna().sum())    # 3061 total
# print(df_sparkling['vintage.wine.taste.structure.tannin'].isna().sum())       # 3069 total

# 2073 ROWS
# Roughly 4/5 with null values
# print(df_rose['vintage.wine.taste.structure.acidity'].isna().sum())      # 1615 total
# print(df_rose['vintage.wine.taste.structure.fizziness'].isna().sum())    # 2073 total
# print(df_rose['vintage.wine.taste.structure.intensity'].isna().sum())    # 1615 total
# print(df_rose['vintage.wine.taste.structure.sweetness'].isna().sum())    # 1615 total
# print(df_rose['vintage.wine.taste.structure.tannin'].isna().sum())       # 2073 total

# 1307 ROWS
# Roughly half with null values
# print(df_dessert['vintage.wine.taste.structure.acidity'].isna().sum())      # 755 total
# print(df_dessert['vintage.wine.taste.structure.fizziness'].isna().sum())    # 1307 total
# print(df_dessert['vintage.wine.taste.structure.intensity'].isna().sum())    # 755 total
# print(df_dessert['vintage.wine.taste.structure.sweetness'].isna().sum())    # 755 total
# print(df_dessert['vintage.wine.taste.structure.tannin'].isna().sum())       # 1300 total

# 1353 ROWS
# Roughly 3/4 with null values
# print(df_fortified['vintage.wine.taste.structure.acidity'].isna().sum())      # 1003 total
# print(df_fortified['vintage.wine.taste.structure.fizziness'].isna().sum())    # 1353 total
# print(df_fortified['vintage.wine.taste.structure.intensity'].isna().sum())    # 1003 total
# print(df_fortified['vintage.wine.taste.structure.sweetness'].isna().sum())    # 1003 total
# print(df_fortified['vintage.wine.taste.structure.tannin'].isna().sum())       # 1351 total
