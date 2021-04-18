from src.data_preprocessing import get_processed_data_from_csv
import pandas as pd


def get_stacked_context_pairs(data):
    context_pairs = ContextPairs()
    pairs = data.apply(context_pairs)
    return pairs.apply(pd.Series).stack().reset_index(drop=True)


class ContextPairs:
    def __init__(self, window_size=2):
        self.window_size = window_size
        self.context_pairs = []

    def __call__(self, text):
        self.context_pairs = []
        for index, word in enumerate(text):
            for window_index in range(self.window_size):
                self.add_context_from_future(text, index, window_index)
                self.add_context_from_past(text, index, window_index)

        return self.context_pairs

    def add_context_from_future(self, text, index, window_index):
        context_index = index + 1 + window_index
        if context_index < len(text):
            self.context_pairs.append([
                text[index],
                text[context_index]
            ])

    def add_context_from_past(self, text, index, window_index):
        context_index = index - (1 + window_index)
        if context_index >= 0:
            self.context_pairs.append([
                text[index],
                text[context_index]
            ])


if __name__ == "__main__":
    processed_data = get_processed_data_from_csv("../data/sample_data.csv")
    print(get_stacked_context_pairs(processed_data))
