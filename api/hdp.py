import random
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

class HDPTopicClassifier:
    def __init__(self, alpha, gamma, seed=None):
        """
        Initialize the HDP Topic Classifier.

        Parameters:
        - alpha: Document-level concentration parameter
        - gamma: Global concentration parameter
        - eta: Symmetric Dirichlet prior for topics
        - seed: Random seed
        """
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        if seed is not None:
            random.seed(seed)  # Set random seed for reproducibility
            np.random.seed(seed)  # Set numpy random seed

    def _initialize(self, documents):
        # decide vocabulary size
        self.vocab_size = len(set(word for doc in documents for word in doc))        
        self.documents = documents
        self.num_docs = len(documents)
        self.word2id = {}  # for internal vocab indexing
        self.id2word = []
        self.word_topic_counts = defaultdict(lambda: np.zeros(self.vocab_size))  # topic-word counts
        self.topic_counts = defaultdict(int)  # total word count per topic
        self.topic_table_counts = defaultdict(int)  # table counts per topic
        self.doc_tables = [defaultdict(list) for _ in documents]  # tables per document
        self.table_to_topic = [{} for _ in documents]  # table-to-topic mapping per doc
        self.topic_id_counter = 0  # unique topic id generator
        self.table_id_counter = [0] * self.num_docs  # unique table id per doc

        self.table_assignments = []  # table assignment for each word in each doc
        self.topic_assignments = []  # topic assignment for each word in each doc

        for j, doc in enumerate(documents):
            tables = []
            topics = []
            for word in doc:
                wid = self._get_word_id(word)  # get or assign word id
                # Decide whether to create a new table or use an existing one
                if self.table_id_counter[j] == 0 or random.random() < self.alpha / (self.alpha + len(self.doc_tables[j])):
                    t = self.table_id_counter[j]
                    self.table_id_counter[j] += 1
                    # Decide whether to create a new topic or use an existing one
                    if len(self.topic_counts) == 0 or random.random() < self.gamma / (self.gamma + len(self.topic_counts)):
                        k = self._new_topic()
                    else:
                        # pick existing topic
                        k = random.choice(list(self.topic_counts.keys()))
                    self.table_to_topic[j][t] = k
                    self.topic_table_counts[k] += 1
                else:
                    t = random.choice(list(self.table_to_topic[j].keys()))  # pick existing table
                    k = self.table_to_topic[j][t]
                self.doc_tables[j][t].append(wid)  # assign word to table
                self.word_topic_counts[k][wid] += 1  # increment word-topic count
                self.topic_counts[k] += 1  # increment topic count
                
                tables.append(t)
                topics.append(k)
            self.table_assignments.append(tables)
            self.topic_assignments.append(topics)

    def _get_word_id(self, word):
        if word not in self.word2id:
            self.word2id[word] = len(self.word2id)  # assign new id
            self.id2word.append(word)
        return self.word2id[word]

    def _new_topic(self):
        k = self.topic_id_counter  # get new topic id
        self.topic_id_counter += 1
        return k

    def fit(self, documents, iterations=100):
        self._initialize(documents)

        for it in tqdm(range(iterations)):
            for j, doc in enumerate(documents):
                for i, word in enumerate(doc):
                    wid = self._get_word_id(word)
                    t_old = self.table_assignments[j][i]
                    k_old = self.table_to_topic[j][t_old]

                    # Decrement counts for current assignment
                    self.doc_tables[j][t_old].remove(wid)
                    self.word_topic_counts[k_old][wid] -= 1
                    self.topic_counts[k_old] -= 1
                    if len(self.doc_tables[j][t_old]) == 0:
                        del self.doc_tables[j][t_old]  # remove empty table
                        del self.table_to_topic[j][t_old] # remove empty table-to-topic mapping
                        self.topic_table_counts[k_old] -= 1
                        # Check if the topic is now unused in this document
                        topic_still_used = any(
                            self.table_to_topic[j][t] == k_old for t in self.table_to_topic[j]
                        )
                        # Check if the topic is unused in all documents
                        topic_used_elsewhere = any(
                            k_old in doc_table_to_topic.values()
                            for idx, doc_table_to_topic in enumerate(self.table_to_topic)
                            if idx != j
                        )
                        if not topic_still_used and not topic_used_elsewhere:
                            # Remove topic if no tables in any document use it
                            if k_old in self.word_topic_counts:
                                del self.word_topic_counts[k_old]
                            if k_old in self.topic_counts:
                                del self.topic_counts[k_old]
                            if k_old in self.topic_table_counts:
                                del self.topic_table_counts[k_old]

                    # Compute probabilities for existing tables
                    table_probs = []
                    table_ids = list(self.doc_tables[j].keys())
                    for t in table_ids:
                        k = self.table_to_topic[j][t]
                        phi_k = self.word_topic_counts[k]
                        prob = len(self.doc_tables[j][t]) * phi_k[wid]/(sum(phi_k))
                        table_probs.append(prob)

                    # Probability of creating a new table (and possibly new topic)
                    new_topic_probs = []
                    for k in self.word_topic_counts:
                        phi_k = self.word_topic_counts[k]
                        prob = (self.topic_table_counts[k] / (sum(self.topic_table_counts.values()) + self.gamma)) * phi_k[wid]/sum(phi_k)
                        new_topic_probs.append(prob)
                    base_prob = self.gamma / (sum(self.topic_table_counts.values()) + self.gamma) * (1.0 / self.vocab_size)
                    new_table_prob = self.alpha * (sum(new_topic_probs) + base_prob)
                    table_probs.append(new_table_prob)
                    new_topic_probs.append(base_prob)
                    
                    table_probs = np.array(table_probs)
                    table_probs /= table_probs.sum()  # normalize
                    t_idx = np.random.choice(len(table_probs), p=table_probs)  # sample table

                    # Assign word to table and topic
                    if t_idx < len(table_ids):
                        t_new = table_ids[t_idx]
                        k_new = self.table_to_topic[j][t_new]
                    else:
                        t_new = self.table_id_counter[j]
                        self.table_id_counter[j] += 1
                        
                        new_topic_probs = np.array(new_topic_probs)
                        new_topic_probs /= new_topic_probs.sum()
                        topic_idx = np.random.choice(len(new_topic_probs), p=new_topic_probs)  # sample topic
                        if topic_idx < len(self.word_topic_counts):
                            k_new = list(self.word_topic_counts.keys())[topic_idx]
                        else:
                            k_new = self._new_topic()
                        self.topic_table_counts[k_new] += 1
                        self.table_to_topic[j][t_new] = k_new
                    self.doc_tables[j][t_new].append(wid)
                    self.word_topic_counts[k_new][wid] += 1
                    self.topic_counts[k_new] += 1
                    self.table_assignments[j][i] = t_new
                    self.topic_assignments[j][i] = k_new

        self._reorder_topics()  # Reorder topics to make them consecutive

    def _reorder_topics(self):
        """
        Give new topic ids to the topics by making them consecutive.
        """
        # Map old topic ids to new consecutive ids
        old_to_new = {}
        new_id = 0
        for old_id in sorted(self.word_topic_counts.keys()):
            old_to_new[old_id] = new_id
            new_id += 1

        # Update word_topic_counts and topic_counts
        self.word_topic_counts = {old_to_new[k]: v for k, v in self.word_topic_counts.items()}
        self.topic_counts = {old_to_new[k]: v for k, v in self.topic_counts.items()}

        # Update table_to_topic mappings
        for j in range(self.num_docs):
            for t in self.table_to_topic[j]:
                old_k = self.table_to_topic[j][t]
                self.table_to_topic[j][t] = old_to_new[old_k]

        # Update topic_assignments
        for j in range(self.num_docs):
            self.topic_assignments[j] = [old_to_new[k] for k in self.topic_assignments[j]]

        # Update topic_id_counter
        self.topic_id_counter = new_id
        

    def get_document_topics(self):
        """
        Estimate document-topic proportions using table-weighted estimation.
        Returns:
        - List of topic proportion vectors, one per document.
        """
        doc_topic_dist = []
        for j in range(self.num_docs):
            topic_counts = Counter()
            total = 0
            for t, words in self.doc_tables[j].items():
                k = self.table_to_topic[j][t]
                count = len(words)
                topic_counts[k] += count
                total += count
            dist = {k: topic_counts[k] / total for k in topic_counts}  # normalize topic counts
            doc_topic_dist.append(dist)
        return doc_topic_dist
    
    def get_topics(self, top_n=10):
        """
        Get the topics and their associated words with probabilities.
        Returns:
        - List of topics, each topic is a list of (word, probability) tuples.
        """
        topics = []
        for k in sorted(self.word_topic_counts.keys()):
            word_counts = self.word_topic_counts[k]
            total = word_counts.sum()
            if total == 0:
                continue
            top_indices = np.argsort(word_counts)[-top_n:][::-1]
            topic_words = [(self.id2word[wid], word_counts[wid] / total) for wid in top_indices]
            topics.append(topic_words)
        return topics