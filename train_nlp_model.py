import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import warnings

warnings.filterwarnings("ignore")

# 🧠 Sample training data
TRAIN_DATA = [
    ("I want a red saree under 1500", {
        "entities": [(10, 13, "COLOR"), (14, 19, "PRODUCT"), (26, 30, "PRICE")]
    }),
    ("Show me blue shoes below 2000", {
        "entities": [(8, 12, "COLOR"), (13, 18, "PRODUCT"), (25, 29, "PRICE")]
    }),
    ("Looking for a green kurti under 1000 rupees", {
        "entities": [(16, 21, "COLOR"), (22, 27, "PRODUCT"), (34, 38, "PRICE")]
    }),
    ("Find yellow tops under 999", {
        "entities": [(5, 11, "COLOR"), (12, 16, "PRODUCT"), (23, 26, "PRICE")]
    }),
    ("Suggest black jeans under 2500", {
        "entities": [(8, 13, "COLOR"), (14, 19, "PRODUCT"), (26, 30, "PRICE")]
    }),
    ("I want a pink dress below 1800", {
        "entities": [(10, 14, "COLOR"), (15, 20, "PRODUCT"), (27, 31, "PRICE")]
    }),
]

# 🆕 Create blank English model
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# 🏷 Add custom entity labels
labels = ["PRODUCT", "COLOR", "PRICE"]
for label in labels:
    ner.add_label(label)

# 🔁 Training loop
optimizer = nlp.begin_training()
n_iter = 30

for itn in range(n_iter):
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        for text, annotations in batch:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(f"Iteration {itn+1}: Losses = {losses}")

# 💾 Save model
output_dir = "./product_filter_nlp"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")
