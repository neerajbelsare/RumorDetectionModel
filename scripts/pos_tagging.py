import nltk
from nltk import word_tokenize, pos_tag



nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def pos_tagging(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Perform POS tagging
    tagged_tokens = pos_tag(tokens)

    return tagged_tokens


def pos_for_dataframe(dataframe, text_column_name):
    # Apply POS tagging to a DataFrame column
    dataframe['POS_Tagged'] = dataframe[text_column_name].apply(pos_tagging)

    return dataframe