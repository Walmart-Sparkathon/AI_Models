import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import warnings
from training_data import TRAIN_DATA  # <-- This line is key

warnings.filterwarnings("ignore")

# Create blank model
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Add labels
labels = ["PRODUCT", "COLOR", "PRICE"]
for label in labels:
    ner.add_label(label)

# Training
optimizer = nlp.begin_training()
for itn in range(30):
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
