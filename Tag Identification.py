import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def extract_tags(text, num_tags=5):
    # Tokenize the text
    words = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Count the frequency of each word
    word_counts = Counter(filtered_words)
    
    # Extract the most common tags
    tags = word_counts.most_common(num_tags)
    
    return [tag[0] for tag in tags]

# Example usage
if __name__ == "__main__":
    sample_text = """
    Python is a powerful programming language. It is widely used for web development, data analysis, artificial intelligence, and scientific computing.
    """
    tags = extract_tags(sample_text)
    print("Extracted Tags:", tags)
