import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

PRICE = "price.amount"
RATING = "vintage.statistics.ratings_average"
FIZZINESS = "vintage.wine.taste.structure.fizziness"
ACIDITY = 'vintage.wine.taste.structure.acidity'
INTENSITY = 'vintage.wine.taste.structure.intensity'
SWEETNESS = 'vintage.wine.taste.structure.sweetness'
TANNIN = 'vintage.wine.taste.structure.tannin'
RATING_CNT = 'vintage.statistics.ratings_count'
RATING_VALID = 'vintage.wine.has_valid_ratings'
ORIGIN_COUNTRY = 'vintage.wine.region.country.name'
ORIGIN_REGION = 'vintage.wine.style.region.name'
YEAR = 'vintage.year'
WINE_TYPE = 'vintage.wine.type_id'
WINE_TYPES = {
    "RED": 1,
    "WHITE": 2,
    "SPARKLING": 3,
    "ROSE": 4,
    "DESERT WINE": 7,
    "FORTIFIED WINE": 24
}
BOTTLE_VOLUME = "price.bottle_type.volume_ml"
REGION = "vintage.wine.style.regional_name"

plt.style.use('seaborn')
df = pd.read_csv("data_processed_2.csv")
df[YEAR] = pd.to_numeric(df[YEAR], errors='coerce')
df_valid = df[df["vintage.statistics.status"] != "BelowThreshold"]


def scatter_plot(df):
    x = df[RATING]
    y = df[PRICE]
    model = make_pipeline(PolynomialFeatures(1), LinearRegression())
    model.fit(np.array(x).reshape(-1, 1), y)
    x_reg = np.arange(0, 5.1, 0.1)
    y_reg = model.predict(x_reg.reshape(-1, 1))

    plt.scatter(x, y, s=3)
    ax = plt.gca()
    plt.plot(x_reg, y_reg, "r-")
    ax.set_ylim([0, 1000])
    ax.set_xlim([0, 5])
    plt.show()


def plot_countries_price(df):
    data = df[["vintage.wine.region.country.name", "vintage.statistics.ratings_average", "price.amount"]]
    data = data.groupby(["vintage.wine.region.country.name"]).agg(
        {"vintage.statistics.ratings_average": ["mean", "count"], "price.amount": "mean"}
    )
    data = data[data[('vintage.statistics.ratings_average', 'count')] > 100]
    column = (PRICE, 'mean')
    data = data.sort_values(by=[column])
    plt.bar(data.index, data[column], color="#4C72B0")
    plt.xticks(rotation=45)
    plt.title("Price x Origin")
    plt.xlabel("Country of origin")
    plt.ylabel("Average price")
    plt.tight_layout()
    plt.bar(data.index, data[(PRICE, 'mean')])
    plt.savefig('Price x Origin.pdf')
    plt.show()


def plot_countries_rating(df):
    data = df[["vintage.wine.region.country.name", "vintage.statistics.ratings_average", "price.amount"]]
    data = data.groupby(["vintage.wine.region.country.name"]).agg(
        {"vintage.statistics.ratings_average": ["mean", "count"], "price.amount": "mean"}
    )
    data = data[data[('vintage.statistics.ratings_average', 'count')] > 100]
    column = ('vintage.statistics.ratings_average', 'mean')
    data = data.sort_values(by=[column])
    plt.bar(data.index, data[column])
    plt.xticks(rotation=45)
    plt.title("Rating x Origin")
    plt.xlabel("Country of origin")
    plt.ylabel("Average rating")
    plt.tight_layout()
    # plt.bar(data.index, data['price.amount'])
    plt.savefig('Rating x Origin.pdf')
    plt.show()


def correlation(observed_cols, df):
    result = pd.DataFrame(columns=observed_cols, index=observed_cols)
    for val1 in observed_cols:
        for val2 in observed_cols:
            result[val1][val2] = df[val1].corr(df[val2])
    return result


def plot_taste_profile(data, profile, y_axis, x_label, y_label, title):
    t_start = 1.
    t_end = 5
    t_interval = 0.5
    data = data[[profile, y_axis]] \
        .groupby(pd.cut(data[profile], np.arange(t_start, t_end + t_interval, t_interval))) \
        .agg({
        profile: "count",
        y_axis: "mean"
    })
    plt.bar([str(x.right) for x in data.index], data[y_axis])
    # plt.bar([str(x.right) for x in data.index], data[profile], width=0.5)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(f'{title}.pdf')
    plt.show()


# data = df[[BOTTLE_VOLUME, RATING, PRICE]].groupby([BOTTLE_VOLUME]).agg({
#     PRICE: ("mean", "count", "var"),
#     RATING: ("mean", "var")
# })
# x = [str(val) for val in data.index]
# y = data[(PRICE, 'count')]
# plt.bar(x, y)
# plt.show()
# red_wines_valid = df_valid[df_valid[WINE_TYPE] == WINE_TYPES["RED"]]
# red_tuscn_valid = red_tuscan[red_tuscan["vintage.statistics.status"] != "BelowThreshold"]
# white_wines = df[df[WINE_TYPE] == WINE_TYPES["WHITE"]]

# print(data1[PRICE].corr(red_wines[INTENSITY], method="pearson"))
# print(data1[PRICE].corr(red_wines[INTENSITY], method="kendall"))


def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues


red_wines = df[df[WINE_TYPE] == WINE_TYPES["RED"]]
red_tuscan = red_wines[red_wines[ORIGIN_REGION] == "Toscana"]
print(len(red_tuscan))
print(red_tuscan[PRICE].corr(red_tuscan[ACIDITY], method="spearman"))
print(red_tuscan[PRICE].corr(red_tuscan[INTENSITY], method="spearman"))
print(red_tuscan[PRICE].corr(red_tuscan[SWEETNESS], method="spearman"))
print(red_tuscan[PRICE].corr(red_tuscan[TANNIN], method="spearman"))
print(red_tuscan[SWEETNESS].corr(red_tuscan[INTENSITY], method="spearman"))
res = (calculate_pvalues(red_tuscan[[ACIDITY, INTENSITY, SWEETNESS, TANNIN, PRICE]]))
print(res)
red_tuscan = red_tuscan[red_tuscan[PRICE] < 1000]
t = red_tuscan[red_tuscan[SWEETNESS] < 2.5]

SIZE = 22
params = {'legend.fontsize': SIZE,
          'axes.labelsize': SIZE,
          'axes.titlesize': SIZE,
          'xtick.labelsize': SIZE,
          'ytick.labelsize': SIZE}
pylab.rcParams.update(params)
plt.scatter(t[SWEETNESS], t[PRICE], s=3)
plt.xlabel("Sweetness")
plt.ylabel("Price (EUR)")
plt.tight_layout()
plt.savefig("sweetness.pdf")
plt.show()
t = red_tuscan[red_tuscan[INTENSITY] > 2.5]
plt.scatter(t[INTENSITY], t[PRICE], s=3)
plt.xlabel("Intensity")
plt.ylabel("Price (EUR)")
plt.tight_layout()
plt.savefig("intensity.pdf")
plt.show()
#
# t_start = 50
# t_end = 100
# t_interval = 1
# data1 = data1[[PRICE, RATING]] \
#     .groupby(pd.cut(data1[PRICE], np.arange(t_start, t_end + t_interval, t_interval))) \
#     .agg({
#     PRICE: "count"
# })
#
# plt.bar([x.right for x in data1.index], data1[PRICE])
# # plt.bar([str(x.right) for x in data.index], data[profile], width=0.5)
# plt.xticks(rotation=45)
# plt.title("Price x Wine count")
# plt.xlabel("Price (EUR)")
# plt.ylabel("Wine count")
# plt.tight_layout()
# plt.savefig("wine count x price.pdf")
# plt.show()
# plot_taste_profile(df, INTENSITY, PRICE, "Acidity", "Average price", "Price x Acidity")
# observed_cols = [PRICE, RATING, RATING_CNT, FIZZINESS, ACIDITY, INTENSITY, SWEETNESS, TANNIN, YEAR]
#
# correlation(observed_cols, red_wines).to_csv("red_wines_corr.csv", index=True)
# correlation(observed_cols, white_wines).to_csv("white_wines_corr.csv", index=True)
#
# plot_countries_rating(df_valid)
# plot_countries_price(df)
#
# start = 2.5
# end = 5
# interval = 0.1
# data1 = df_valid[[RATING, PRICE]] \
#     .groupby(pd.cut(df_valid[RATING], np.arange(start, end + interval, interval))) \
#     .agg({
#     RATING: "count",
#     PRICE: "mean"
# })
# plt.bar([str(x.right) for x in data1.index], data1[RATING])
# plt.xticks(rotation=45)
# plt.title("Rating distribution")
# plt.xlabel("Rating")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.savefig('Rating distribution')
# plt.show()
#
# plt.bar([str(x.right) for x in data1.index], data1[PRICE])
# plt.xticks(rotation=45)
# plt.title("Price x Rating")
# plt.xlabel("Rating")
# plt.ylabel("Average price")
# plt.tight_layout()
# plt.savefig('Price x Rating')
# plt.show()
#
# data2 = df[df[YEAR] > 2000]
# data2 = data2[[YEAR, PRICE]].groupby(YEAR).mean()
#
# plt.bar(data2.index, data2[PRICE])
#
# plt.xticks(rotation=45)
# plt.title("Price x Year")
# plt.xlabel("Year")
# plt.ylabel("Average price")
# plt.tight_layout()
# plt.savefig('Price x Year')
# plt.show()
#
#
#
# # plot_taste_profile(df_valid, ACIDITY, RATING, "Acidity", "Average rating", "Rating x Acidity (valid ratings only)")
# # plot_taste_profile(df_valid, INTENSITY, RATING, "Intensity", "Average rating",
# #                    "Rating x Intensity (valid ratings only)")
# # plot_taste_profile(df_valid, TANNIN, RATING, "Tannin", "Average rating", "Rating x Tannin (valid ratings only)")
# # plot_taste_profile(df_valid, SWEETNESS, RATING, "Sweetness", "Average rating",
# #                    "Rating x Sweetness (valid ratings only)")
# #
# # plot_taste_profile(df, ACIDITY, PRICE, "Acidity", "Average price", "Price x Acidity")
# # plot_taste_profile(df, INTENSITY, PRICE, "Intensity", "Average price", "Price x Intensity")
# # plot_taste_profile(df, TANNIN, PRICE, "Tannin", "Average price", "Price x Tannin")
# # plot_taste_profile(df, SWEETNESS, PRICE, "Sweetness", "Average price", "Price x Sweetness")
#
# plot_taste_profile(df, ACIDITY, PRICE, "Acidity", "Average price",
#                    "Price x Acidity")
# plot_taste_profile(df, INTENSITY, PRICE, "Intensity", "Average price",
#                    "Price x Intensity")
# plot_taste_profile(df, TANNIN, PRICE, "Tannin", "Average price",
#                    "Price x Tannin")
# plot_taste_profile(df, SWEETNESS, PRICE, "Sweetness", "Average price",
#                    "Price x Sweetness")
#
# plot_taste_profile(red_wines, ACIDITY, PRICE, "Acidity", "Average price",
#                    "Red wines: Price x Acidity")
# plot_taste_profile(red_wines, INTENSITY, PRICE, "Intensity", "Average price",
#                    "Red wines: Price x Intensity")
# plot_taste_profile(red_wines, TANNIN, PRICE, "Tannin", "Average price",
#                    "Red wines: Price x Tannin")
# plot_taste_profile(red_wines, SWEETNESS, PRICE, "Sweetness", "Average price",
#                    "Red wines: Price x Sweetness")
#
# plot_taste_profile(white_wines, ACIDITY, PRICE, "Acidity", "Average price",
#                    "White wines: Price x Acidity")
# plot_taste_profile(white_wines, INTENSITY, PRICE, "Intensity", "Average price",
#                    "White wines: Price x Intensity")
# plot_taste_profile(white_wines, SWEETNESS, PRICE, "Sweetness", "Average price",
#                    "White wines: Price x Sweetness")
