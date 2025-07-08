import logging
import numpy as np
from api.hdp import HDPTopicClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_hdp_classifier():
    # Example documents
    # Repeat same words for same topics repeatedly to strengthen topic signals
    fruit_words = ["apple", "banana", "orange", ]
    veg_words = ["lettuce", "carrot", "broccoli", ]
    meat_words = ["beef", "chicken", "pork",]
    kitchen_words = ["knife", "cutting", "board", ]

    documents = []
    documents.append(fruit_words * 200)
    documents.append(fruit_words * 200)
    documents.append(veg_words * 200)
    documents.append(veg_words * 200)
    documents.append(meat_words * 200)
    documents.append(meat_words * 200)
    documents.append(kitchen_words * 200)
    documents.append(kitchen_words * 200)
    # Add mixed topic documents
    documents.append(fruit_words * 100 + veg_words * 100)
    documents.append(meat_words * 100 + kitchen_words * 100)
    documents.append(fruit_words * 50 + veg_words * 50 + meat_words * 50 + kitchen_words * 50)

    hdp = HDPTopicClassifier(alpha=0.1, gamma=0.1, seed=0)
    hdp.fit(documents, iterations=100)
    doc_topics = hdp.get_document_topics()

    logging.info("Document Topics:")
    for i, doc in enumerate(doc_topics):
        formatted = ", ".join([f"Topic {k}: {prob:.3f}" for k, prob in doc.items()])
        logging.info(f"Document {i}: {formatted}")

    topics = hdp.get_topics()
    logging.info("Topics:")
    for i, topic in enumerate(topics):
        formatted = ", ".join([f"{word}: {prob:.3f}" for word, prob in topic if prob > 0])
        logging.info(f"Topic {i}: {formatted}")

def run_tests():
    logging.info("Running HDP Classifier Tests")
    test_hdp_classifier()
    logging.info("HDP Classifier Tests Completed")

if __name__ == "__main__":
    run_tests()
    logging.info("All tests completed.")