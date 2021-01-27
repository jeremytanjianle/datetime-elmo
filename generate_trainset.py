"""
Generates synthetic dataset
TODO: 
1. Add "the 25th of december"
"""
import datetime
import random
import pandas as pd

decodable_date = lambda datetime_obj: datetime_obj.strftime("%d/%m/%Y")

single_month_abbriev_comma = lambda date_time: date_time.strftime("%d %b %y")
single_month_abbriev = lambda date_time: date_time.strftime("%d %b %y")
single_month_full_comma = lambda date_time: date_time.strftime("%d %B %y")
single_month_full = lambda date_time: date_time.strftime("%d %B %y")
single_date_digit_slashed = lambda date_time:date_time.strftime("%d/%m/%Y")
single_date_digit_dashed = lambda date_time:date_time.strftime("%d-%m-%Y")
single_month_first = lambda date_time: date_time.strftime("%B %d %y")
single_abbriev_month_first = lambda date_time: date_time.strftime("%b %d %y")

double_month_abbriev_comma = lambda date_time, end_date: same_or_diff_year(date_time, end_date, "%d %b %y") 
double_month_abbriev = lambda date_time, end_date: same_or_diff_year(date_time, end_date, "%d %b %y") 
double_month_full_comma = lambda date_time, end_date: same_or_diff_year(date_time, end_date, "%d %B %y") 
double_month_full = lambda date_time, end_date: same_or_diff_year(date_time, end_date, "%d %B %y") 
double_date_digit_slashed = lambda date_time, end_date: same_or_diff_year(date_time, end_date, "%d/%m/%Y") 
double_date_digit_dashed = lambda date_time, end_date: same_or_diff_year(date_time, end_date, "%d-%m-%Y") 

def remove_padding(datestring):
    if datestring.startswith('0') & (random.uniform(0,1)<0.8):
        return datestring[1:]
    else:
        return datestring

single_date_to_string = [single_month_abbriev_comma,
                         single_month_abbriev, 
                         single_month_full_comma, 
                         single_month_full, 
                         single_date_digit_slashed, 
                         single_date_digit_dashed,
                         single_month_first,
                         single_abbriev_month_first
                         ]

double_date_to_string = [double_month_abbriev_comma,
                         double_month_abbriev, 
                         double_month_full_comma, 
                         double_month_full, 
                         double_date_digit_slashed, 
                         double_date_digit_dashed]

def random_date(start_date = datetime.datetime(1990, 1, 1), end_date=datetime.datetime.now(), same_year_end=False):
    """
    Generate random datetime
    """
    if same_year_end:
        end_date.replace(year = start_date.year)
    time_between_dates = end_date - start_date

    days_between_dates = max(time_between_dates.days,1)
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    
    random_sec = random.randrange(86400)
    random_date = random_date + datetime.timedelta(seconds=random_sec)
    
    return random_date

def same_or_diff_year(date_time, end_date, format_string):
    if (date_time.year == end_date.year) & (random.uniform(0,1)<0.9):
        remove_year_strin = format_string.replace('%y','').replace('%Y','').replace('/%Y','').strip()
        return date_time.strftime(remove_year_strin) + ' - ' + end_date.strftime(format_string)
    else:
        return date_time.strftime(format_string) + ' - ' + end_date.strftime(format_string)
    
def generate_natural_date_string():    
    """
    workhorse function: generates the dates and datestrings
    """
    start_date = random_date()

    if (random.uniform(0,1)<0.5):
        # generate single date
        end_date = ''
        string_generator = random.choice(single_date_to_string)
        date_string = string_generator(start_date)
    else:
        # generate two dates
        end_date = random_date(start_date=start_date, same_year_end = random.uniform(0,1)<0.9)
        string_generator = random.choice(double_date_to_string)
        date_string = string_generator(start_date, end_date)
        
    if (random.uniform(0,1)<0.2):
        date_string = date_string.lower()

    return remove_padding(date_string), start_date, end_date

if __name__ == "__main__":
    len_obs = 100000
    rows = []
    for i in range(len_obs):
        
        # generate dates and input output strings
        date_string, start_date, end_date = generate_natural_date_string()
        decodable_string = decodable_date(start_date)
        if end_date != '': decodable_string = decodable_string + ' - ' + decodable_date(end_date)
            
        row = [start_date, end_date, date_string, decodable_string]
        rows.append(row)
        
    trainset = pd.DataFrame(rows, columns = ["start_date", "end_date", "date_string", "decodable_string"])
    trainset.to_csv("synthetic_datestrings.csv")