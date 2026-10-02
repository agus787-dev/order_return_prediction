import sys

sys.path.append("..")
from src.offline_testing import OfflineTester


tester = OfflineTester()

# an order
raw = {
    "item_price": 20, "item_size": "M", "item_color": "blue",
    "brand_id": 12, "user_title": "Mr", "user_state": "Berlin",
    "order_date": "2016-03-10", "user_reg_date": "2015-01-05",
    "user_dob": "1990-07-21", "delivery_date": "2016-03-10",
}


tester.explain(raw)

# probs = tester.predict_proba(df_new)

# for i, p in enumerate(probs):
#     print(f"Order {i}: {p*100:.0f}% returned, {(1-p)*100:.0f}% not returned")
# print(tester.predict_proba(raw))   # masalan [0.37]
# print(tester.predict(raw))         # [0] yoki [1]
