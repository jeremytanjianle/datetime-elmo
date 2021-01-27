import pandas as pd
from src.use_parser import natural_datetime_use

if __name__ == '__main__':
    parser = natural_datetime_use()
    parser.train_on_synthetic_data(epochs=10)
    parser.save()