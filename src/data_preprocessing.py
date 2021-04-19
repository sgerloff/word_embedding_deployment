import pandas as pd
import gensim, re


class Preprocessor:
    def __init__(self):
        # Replace usernames: @KlausHeinz -> __username__
        self.usernames = re.compile("@\w*")
        # Replace urls: http://this.is.a.url.de/adsf -> __url__
        self.urls = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        # Replace hashtag: #COVID -> __hashtag__COVID
        self.hashtags = re.compile("#+?(?=\w*)")

    def replace_twitter_tags(self, text):
        text = self.usernames.sub("__username__", text)
        text = self.hashtags.sub("__hashtag__", text)
        return text

    def replace_urls(self, text):
        return self.urls.sub("__url__", text)

    @staticmethod
    def pretokenize(text):
        """
        This function helps with words like "it's" and "don't".
        """
        text = gensim.utils.tokenize(text)
        return " ".join(text)

    @staticmethod
    def clean_text(text):
        text = gensim.parsing.preprocessing.remove_stopwords(text)
        text = gensim.parsing.preprocessing.strip_short(text)
        text = gensim.parsing.preprocessing.strip_numeric(text)
        return text.lower()

    def __call__(self, text):
        text = self.replace_twitter_tags(text)
        text = self.replace_urls(text)
        text = self.pretokenize(text)
        text = self.clean_text(text)
        return text

    def transform_bytes(self, text):
        print(text, type(text))
        text = text.numpy().decode("utf-8")
        text = self(text)
        return text.encode("utf-8")



def get_processed_data_from_csv(path_to_csv):
    raw_data = pd.read_csv(path_to_csv)
    preprocess = Preprocessor()
    raw_data["OriginalTweet"] = raw_data["OriginalTweet"].apply(preprocess)
    return raw_data


if __name__ == "__main__":
    processed_data = get_processed_data_from_csv("../data/sample_data.csv")
    print(processed_data["OriginalTweet"])
