import json

PATH = "data/raw/amazon_metadata_lookup.json"

with open(PATH, "r", encoding="utf-8") as f:
    products = json.load(f)

print("Total products:", len(products))


def get_num(value):
    try:
        return float(value)
    except:
        return None


# --------------------------------------------------
# TEST 1 — Normal / high-review product
# --------------------------------------------------

print("\n========== TEST 1: NORMAL PRODUCT ==========")

count = 0

for asin, p in products.items():

    rating = get_num(p.get("average_rating"))
    reviews = get_num(p.get("rating_number"))
    seller = p.get("store")
    title = p.get("title")
    description = p.get("description")

    if (
        rating is not None
        and rating >= 4
        and reviews is not None
        and reviews >= 100
        and seller
        and title
        and description
    ):
        print(
            asin,
            "| rating:", rating,
            "| reviews:", reviews,
            "| seller:", seller,
            "| title:", str(title)[:70]
        )

        count += 1

        if count == 3:
            break


# --------------------------------------------------
# TEST 2 — Very low reviews
# --------------------------------------------------

print("\n========== TEST 2: VERY LOW REVIEWS ==========")

count = 0

for asin, p in products.items():

    reviews = get_num(p.get("rating_number"))

    if reviews is not None and reviews <= 5:

        print(
            asin,
            "| rating:", p.get("average_rating"),
            "| reviews:", reviews,
            "| seller:", p.get("store"),
            "| title:", str(p.get("title"))[:70]
        )

        count += 1

        if count == 3:
            break


# --------------------------------------------------
# TEST 3 — High rating + few reviews
# --------------------------------------------------

print("\n========== TEST 3: HIGH RATING + FEW REVIEWS ==========")

count = 0

for asin, p in products.items():

    rating = get_num(p.get("average_rating"))
    reviews = get_num(p.get("rating_number"))

    if (
        rating is not None
        and rating >= 4.5
        and reviews is not None
        and reviews <= 10
    ):

        print(
            asin,
            "| rating:", rating,
            "| reviews:", reviews,
            "| seller:", p.get("store"),
            "| title:", str(p.get("title"))[:70]
        )

        count += 1

        if count == 3:
            break


# --------------------------------------------------
# TEST 4 — Missing seller
# --------------------------------------------------

print("\n========== TEST 4: MISSING SELLER ==========")

count = 0

for asin, p in products.items():

    seller = p.get("store")

    if seller is None or str(seller).strip() == "":

        print(
            asin,
            "| rating:", p.get("average_rating"),
            "| reviews:", p.get("rating_number"),
            "| seller:", repr(seller),
            "| title:", str(p.get("title"))[:70]
        )

        count += 1

        if count == 3:
            break


# --------------------------------------------------
# TEST 5 — Very short title / description
# --------------------------------------------------

print("\n========== TEST 5: SHORT METADATA ==========")

count = 0

for asin, p in products.items():

    title = p.get("title")
    description = p.get("description")

    title = "" if title is None else str(title)
    description = "" if description is None else str(description)

    if (
        len(title.strip()) < 10
        and len(description.strip()) < 20
    ):

        print(
            asin,
            "| title length:", len(title.strip()),
            "| description length:", len(description.strip()),
            "| seller:", p.get("store"),
            "| title:", repr(title)
        )

        count += 1

        if count == 3:
            break