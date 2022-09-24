URL = 'https://www.vivino.com/api/explore/explore'

PARAMS = {
    "country_code": "DE",
    "currency_code": "EUR",
    "grape_filter": "varietal",
    "order_by": "price",
    "order": "asc",
    "language": "en",
}

HEADERS = {
    "accept": "application/json",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "cs-CZ,cs;q=0.9,en;q=0.8,sk;q=0.7",
    "content-type": "application/json",
    "cookie": "first_time_visit=RIl%2BK9jqqzI0NaNPdVF4mTWJsMbBLMdQX664qLyz7Y4xeMXLvM3mbjMaoSPmwlNf3xHpRRMNKcp0E6oz5IUrdQefdiZaxappLjTO4ej%2BaroPBiI0yDd8dzh2Ka0rHQ%3D%3D--6YMrfZxIaBH81OiP--CH1bmwt1sfpeJDzFwiRCcw%3D%3D; anonymous_tracking_id=HSi90B4cENZifwTj8K%2FikXHV6IWaTAeknfU%2BYYIKbyI1SiP8DbAZgZjgqjfZLsmcvYav9es2rwlzsWqSgsDJ%2FdgDNbgTgNge6EkW0Xsix%2FlQxyj5CJ%2BSCGXlXBrAKYJXU1Bt%2BoAI8d72UmlnpEXSQ%2FbQ37TdLw86VD2G4sRKAuqN3TdTBwwVYvJv6KJRajEG82FwGa%2BsdASnOsRtB1s3--%2FttR7bRfziGlGkEG--IM0u%2FyWy03%2FBkG8VRCb%2Bmg%3D%3D; _gcl_aw=GCL.1662538133.Cj0KCQjwguGYBhDRARIsAHgRm489-w00FyGCGDndnNrp828HSCBUE1o4RKzgAhIlcZ2IxfMIaU91C0gaAuC7EALw_wcB; _gcl_au=1.1.1861888286.1662538133; _ga=GA1.2.703996242.1662538133; _pin_unauth=dWlkPU9UWXpOV1F3WmpndE5XRTJOeTAwWXpjMkxUazVNRFF0WVRrNU5UZ3hOamt5TldaaQ; _gac_UA-222490-2=1.1662538135.Cj0KCQjwguGYBhDRARIsAHgRm489-w00FyGCGDndnNrp828HSCBUE1o4RKzgAhIlcZ2IxfMIaU91C0gaAuC7EALw_wcB; _hjSessionUser_1506979=eyJpZCI6IjEzNzhlZTAyLTQwM2YtNTU1YS05NzIyLWNlNDNmOWYxMzY5ZiIsImNyZWF0ZWQiOjE2NjI1MzgxMzM0MDMsImV4aXN0aW5nIjp0cnVlfQ==; _gid=GA1.2.1907935859.1663158372; _clck=1q8oc4f|1|f4v|0; eeny_meeny_personalized_upsell_module_v2=6HRz959EuohJQIrFuYFlKhL0vIsiOWvZefDKkWCs6uAWvU3db9MVd0gKqhlXBr1tfFrTMvfZV6iaAn8uB92qaw%3D%3D; eeny_meeny_personalized_cross_sell_v2=AoQsUhORMEbU79zxOYb%2BMsjIT6vZbkcpKeKQEs3Pf44EbHvuljlxr7njr9bDxRpjNaF2HusMXr9LwVcgsXG39Q%3D%3D; _hjSession_1506979=eyJpZCI6IjBjMGYxNDdlLTU4NjYtNDBiZC05MDBjLTAzMDAzYTBmMjM3OSIsImNyZWF0ZWQiOjE2NjMxNzQ5NTQxNDcsImluU2FtcGxlIjpmYWxzZX0=; _hp2_ses_props.3503103446=%7B%22r%22%3A%22https%3A%2F%2Fwww.google.com%2F%22%2C%22ts%22%3A1663174954025%2C%22d%22%3A%22www.vivino.com%22%2C%22h%22%3A%22%2Fterms%22%7D; __stripe_mid=533b1880-b983-4091-99b8-2b0e8a1daeaa05626d; client_cache_key=nHUQjgr5kNcua1RWcLpPiWJsVTaV0rdXyqkHsmo64iQBvNdETaf4kOn%2BURglhPVECwosD57tcYTGK9WwJi1t9a6X2ec0jsraDd824rlDuJuT9oI68gP1HBHUiViCOg2N8RbceJu6m%2FVnqiAyDZOBFHnhioPQHsyYAr9EO04L--4hWPyl6c8Y2%2B55NN--kUE4NnEeUWg%2BdtUPeDgsMQ%3D%3D; _hp2_id.3503103446=%7B%22userId%22%3A%221307251794259415%22%2C%22pageviewId%22%3A%223805146899652780%22%2C%22sessionId%22%3A%224708957596191812%22%2C%22identity%22%3Anull%2C%22trackerVersion%22%3A%224.0%22%7D; cto_bundle=epFDTF9IaE5PbXQzalE1QlpIeHFlRHo0eXBHOG50STZXdW01VUNUU2VRRHhjUUdxSk5RRDNYVG5ITzN0bW9Kc3dFYTdEcXFDT3FjNmdZQXhtT1UlMkZ4R2dSUkRXZ1dGRVpNem1ua0glMkY1U2VrSjZTcEhreW9RbmNHMjdPYzZ6WkRIU1cwWWQ0YkJSckRrR0JXSCUyRlAxM2FLJTJGcTdsUSUzRCUzRA; _derived_epik=dj0yJnU9UEhISUltT09pSldpU0RLUHFNVG1iZ3JxZzZCcTlYa3Mmbj1jSGRRaHRhanJWd2NMVldPb1VLdkt3Jm09MSZ0PUFBQUFBR01pRjZzJnJtPTEmcnQ9QUFBQUFHTWlGNnM; _uetsid=6b4a4ea0342811ed8fe1bdaf0e2025ce; _uetvid=50434ac02e8411eda386db663fd6e1de; _clsk=l4nl2g|1663178671432|99|1|b.clarity.ms/collect; _ruby-web_session=Pcjcl%2B4EKMPbR7ApvYLH%2FOQwn4%2FpcAhrJyCB%2BAu1OLSiKgpnOb1RAISoyWVSwkZH6%2BPkRMufWAlbDkcaQVH7TRHrzbfbPDuhGY4CIs2kPzbjgNaqYGPVtC8vke4gmemjD%2Brx3RQV%2B6sd5OTlU3IC3w%2BHx5IiIFvZeiRKN9AFT%2BmFmONYcyzl7weQgJ5rVH1kud8XhAkQ6vwNdTYH8pMIXa6uMBK%2BYWUTc3ywT7cZ1BRzHCytUiQy3w0%2BFpD8bDUypG9Wazxv78O3qIk7KZWC6Wl2Uq2r1R2pru%2FEIQEyw2jr7PnYZt3O13u2dU06gYesl5u%2FVRmjQ%2BCOZm5fUpa9hqKfmkmJYR0ca6swV%2BJqATJEvdwDksfFw3Vw7mrLcrO%2Bip%2FmybGMJavgb%2FHUrEa%2Bo%2BMKjGLYygRp00ZQ0X5H3p1qXvGkZ%2FJEb0jjgC%2Ba0binXBrgzBBwuYsAzg%2FyrRx%2Bz3lDCBH6G9mhk%2Bhe4DCCUH%2Fub64JPUfTjfy9l1%2B6QDTvuxlOSfwZeWDIK9KwTP5vWhTEhn3fgbsEf9a6DgUc3rHiVIjIvv5HF5V9Rp%2FqmMNNNxHv09iHgoM3yg9JJ0IT%2FyiagrtmXt6Jdl2h0h7%2B0pjP6E%2FWosVxxPvr4%2FHG2xxy23Kou0glZMSxV9dBQqjrBpfpKSau5AcADzPAT8bQ3hDbsOKPMwtxuNkIr894sIWsdmDX4LHZ%2FbRPRmykdVCWTt8aYozYtl54io%2B2nWz4iWG3aPh8fm9%2FpBHbPGWHH1Cb1XdyO0duKQ%3D%3D--UXcxpae5YIRJDk4U--CK2uM7ElrEiuBiOe1vQwVg%3D%3D",
    "dnt": "1",
    "referer": "https://www.vivino.com/explore?e=eJzLLbI1VMvNzLM1V8tNrLA1M1BLrrR1cVVLtnUNDVIrAEqmp9mWJRZlppYk5qjlF6XYpqQWJ6vlJ1XaFhRlJqcCAI6vFUg%3D",
    "sec-ch-ua": "\"Google Chrome\";v=\"105\", \"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"105\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest",
}

PARAM_PAGE = "page"
PARAM_PRICE_MAX = "price_range_max"
PARAM_PRICE_MIN = "price_range_min"
PARAM_WINE_TYPE = r"wine_type_ids[]"
PARAM_COUNTRY_CODE = "country_code"
VALUE_WINE_TYPES = [1, 2, 3, 4, 7, 24]

COLUMNS_OF_INTEREST = [
    "prices",
    "vintage.id",
    "vintage.seo_name",
    "vintage.name",
    "vintage.statistics.status",
    "vintage.statistics.ratings_count",
    "vintage.statistics.ratings_average",
    "vintage.statistics.labels_count",
    "vintage.statistics.wine_ratings_count",
    "vintage.statistics.wine_ratings_average",
    "vintage.statistics.wine_status",
    "vintage.image.location",
    "vintage.image.variations.bottle_large",
    "vintage.image.variations.bottle_medium",
    "vintage.image.variations.bottle_medium_square",
    "vintage.image.variations.bottle_small",
    "vintage.image.variations.bottle_small_square",
    "vintage.image.variations.label",
    "vintage.image.variations.label_large",
    "vintage.image.variations.label_medium",
    "vintage.image.variations.label_medium_square",
    "vintage.image.variations.label_small_square",
    "vintage.image.variations.large",
    "vintage.image.variations.medium",
    "vintage.image.variations.medium_square",
    "vintage.image.variations.small_square",
    "vintage.wine.id",
    "vintage.wine.name",
    "vintage.wine.seo_name",
    "vintage.wine.type_id",
    "vintage.wine.vintage_type",
    "vintage.wine.is_natural",
    "vintage.wine.region.id",
    "vintage.wine.region.name",
    "vintage.wine.region.name_en",
    "vintage.wine.region.seo_name",
    "vintage.wine.region.country.code",
    "vintage.wine.region.country.name",
    "vintage.wine.region.country.native_name",
    "vintage.wine.region.country.seo_name",
    "vintage.wine.region.country.currency.code",
    "vintage.wine.region.country.currency.name",
    "vintage.wine.region.country.currency.prefix",
    "vintage.wine.region.country.currency.suffix",
    "vintage.wine.region.country.regions_count",
    "vintage.wine.region.country.users_count",
    "vintage.wine.region.country.wines_count",
    "vintage.wine.region.country.wineries_count",
    "vintage.wine.region.country.most_used_grapes",
    "vintage.wine.region.background_image",
    "vintage.wine.winery.id",
    "vintage.wine.winery.name",
    "vintage.wine.winery.seo_name",
    "vintage.wine.winery.status",
    "vintage.wine.winery.background_image",
    "vintage.wine.taste.structure.acidity",
    "vintage.wine.taste.structure.fizziness",
    "vintage.wine.taste.structure.intensity",
    "vintage.wine.taste.structure.sweetness",
    "vintage.wine.taste.structure.tannin",
    "vintage.wine.taste.structure.user_structure_count",
    "vintage.wine.taste.structure.calculated_structure_count",
    "vintage.wine.taste.flavor",
    "vintage.wine.statistics.status",
    "vintage.wine.statistics.ratings_count",
    "vintage.wine.statistics.ratings_average",
    "vintage.wine.statistics.labels_count",
    "vintage.wine.statistics.vintages_count",
    "vintage.wine.style.id",
    "vintage.wine.style.seo_name",
    "vintage.wine.style.regional_name",
    "vintage.wine.style.varietal_name",
    "vintage.wine.style.name",
    "vintage.wine.style.image",
    "vintage.wine.style.background_image",
    "vintage.wine.style.description",
    "vintage.wine.style.blurb",
    "vintage.wine.style.interesting_facts",
    "vintage.wine.style.body",
    "vintage.wine.style.body_description",
    "vintage.wine.style.acidity",
    "vintage.wine.style.acidity_description",
    "vintage.wine.style.country.code",
    "vintage.wine.style.country.name",
    "vintage.wine.style.country.native_name",
    "vintage.wine.style.country.seo_name",
    "vintage.wine.style.country.currency.code",
    "vintage.wine.style.country.currency.name",
    "vintage.wine.style.country.currency.prefix",
    "vintage.wine.style.country.currency.suffix",
    "vintage.wine.style.country.regions_count",
    "vintage.wine.style.country.users_count",
    "vintage.wine.style.country.wines_count",
    "vintage.wine.style.country.wineries_count",
    "vintage.wine.style.country.most_used_grapes",
    "vintage.wine.style.wine_type_id",
    "vintage.wine.style.food",
    "vintage.wine.style.grapes",
    "vintage.wine.style.region.id",
    "vintage.wine.style.region.name",
    "vintage.wine.style.region.name_en",
    "vintage.wine.style.region.seo_name",
    "vintage.wine.style.region.country.code",
    "vintage.wine.style.region.country.name",
    "vintage.wine.style.region.country.native_name",
    "vintage.wine.style.region.country.seo_name",
    "vintage.wine.style.region.country.currency.code",
    "vintage.wine.style.region.country.currency.name",
    "vintage.wine.style.region.country.currency.prefix",
    "vintage.wine.style.region.country.currency.suffix",
    "vintage.wine.style.region.country.regions_count",
    "vintage.wine.style.region.country.users_count",
    "vintage.wine.style.region.country.wines_count",
    "vintage.wine.style.region.country.wineries_count",
    "vintage.wine.style.region.country.most_used_grapes",
    "vintage.wine.style.region.parent_id",
    "vintage.wine.style.region.background_image.location",
    "vintage.wine.style.region.background_image.variations.large",
    "vintage.wine.style.region.background_image.variations.medium",
    "vintage.wine.style.region.statistics.wineries_count",
    "vintage.wine.style.region.statistics.wines_count",
    "vintage.wine.style.region.statistics.sub_regions_count",
    "vintage.wine.style.region.statistics.parent_regions_count",
    "vintage.wine.has_valid_ratings",
    "vintage.year",
    "vintage.grapes",
    "vintage.has_valid_ratings",
    "price.id",
    "price.amount",
    "price.discounted_from",
    "price.discount_percent",
    "price.type",
    "price.sku",
    "price.url",
    "price.visibility",
    "price.bottle_type_id",
    "price.currency.code",
    "price.currency.name",
    "price.currency.prefix",
    "price.currency.suffix",
    "price.bottle_type.id",
    "price.bottle_type.name",
    "price.bottle_type.short_name",
    "price.bottle_type.short_name_plural",
    "price.bottle_type.volume_ml",
    "vintage.wine.region.background_image.location",
    "vintage.wine.region.background_image.variations.large",
    "vintage.wine.region.background_image.variations.medium",
    "vintage.wine.winery.background_image.location",
    "vintage.wine.winery.background_image.variations.large",
    "vintage.wine.winery.background_image.variations.medium",
    "vintage.wine.winery.background_image.variations.small",
    "vintage.wine.style.background_image.location",
    "vintage.wine.style.background_image.variations.small",
    "vintage.top_list_rankings",
]