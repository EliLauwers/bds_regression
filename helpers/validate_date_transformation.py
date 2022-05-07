import numpy as np

def validate_date_transformation(raw_dates, transformed_dates):
    response = ""
    for i, (joined_date, trans_date) in enumerate(zip(raw_dates, transformed_dates)):
        all_good = True
        if joined_date != str(trans_date):
            all_good = False
        if type(trans_date) == "float":
            if np.isnan(trans_date):
                all_good = False
        if not all_good:
            response += f"r {i}, joined: {joined_date}, trans: {trans_date}\n"

    with open('logs/validate_date_transformation.txt', 'w') as f:
        f.write(response)
