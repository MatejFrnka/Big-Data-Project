import numpy as np
import pandas as pd

df = pd.read_csv('data_processed.csv')

df2 = df[['vintage.id', 'vintage.wine.id', 'vintage.name', 'vintage.wine.name', 'vintage.statistics.wine_ratings_average',
'vintage.wine.region.name', 'vintage.wine.region.country.name', 'vintage.wine.style.body_description', 'vintage.wine.style.name',
'vintage.wine.type_id', 'vintage.wine.taste.structure.acidity', 'vintage.wine.taste.structure.fizziness', 'vintage.wine.taste.structure.intensity',
'vintage.wine.taste.structure.sweetness', 'vintage.wine.taste.structure.tannin', 'vintage.wine.has_valid_ratings',
'vintage.year', 'price.amount', 'price.bottle_type.volume_ml', 'price_category']].copy()

df2.to_csv('trimmed_data_2.csv', index=False)

print(df2.columns.values)

# 'vintage.id'
# 'vintage.wine.id'
# 'vintage.name'
# 'vintage.wine.name'
# 'vintage.statistics.wine_ratings_average'
# 'vintage.wine.region.name'
# 'vintage.wine.region.country.name'
# 'vintage.wine.style.body_description'
# 'vintage.wine.style.name'     "Style" of wine
# 'vintage.wine.type_id'        Think this is just different integer if red/white/whatever
# 'vintage.wine.taste.structure.acidity'
# 'vintage.wine.taste.structure.fizziness'
# 'vintage.wine.taste.structure.intensity'
# 'vintage.wine.taste.structure.sweetness'
# 'vintage.wine.taste.structure.tannin
# 'vintage.wine.taste.flavor'       This one is interesting looks like lists of flavors for each wine or something
# 'vintage.wine.has_valid_ratings'      Bools
# 'vintage.year'
# 'price.amount'
# 'price.bottle_type.volume_ml'
# 'price_category'
