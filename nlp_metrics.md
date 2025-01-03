The **TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure used to evaluate how important a word is in a document relative to a collection of documents (corpus). It combines two metrics: **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.

### **TF-IDF Formula**:

\[
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t, D)
\]

Where:
- \( t \) is the term (word) being evaluated.
- \( d \) is the document containing the term.
- \( D \) is the collection of all documents.

### **Term Frequency (TF)**:
\[
\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
\]
- This calculates how frequently a word appears in a specific document. The more a word appears in a document, the higher its TF score.

### **Inverse Document Frequency (IDF)**:
\[
\text{IDF}(t, D) = \log \left( \frac{\text{Total number of documents in corpus}}{\text{Number of documents containing term } t} \right)
\]
- The IDF measures how unique or rare a word is across all documents in the corpus. If a word appears in many documents, the IDF is low; if it appears in fewer documents, the IDF is high.

### **Example**:
Suppose the word "security" appears 5 times in a document containing 100 words (TF), and it appears in 10 out of 1000 documents in the corpus (IDF).

- **TF** = \( \frac{5}{100} = 0.05 \)
- **IDF** = \( \log \left( \frac{1000}{10} \right) = \log(100) = 2 \)

Thus, **TF-IDF** for "security" = \( 0.05 \times 2 = 0.1 \).

### **Interpretation**:
- **Higher TF-IDF**: The term is important in the specific document but rare across the corpus.
- **Lower TF-IDF**: The term either appears rarely in the document or is very common across many documents.

TF-IDF is widely used in search engines, text mining, and recommendation systems to rank document relevance based on query terms.
