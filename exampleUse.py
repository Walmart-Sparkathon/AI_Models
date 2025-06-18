import spacy

# Load trained model
nlp = spacy.load("./product_filter_nlp")

# Sample input
text = "I need a red top "
doc = nlp(text)

# Initialize fields
product = None
color = None
price = None

# Extract entities
for ent in doc.ents:
    if ent.label_ == "PRODUCT":
        product = ent.text
    elif ent.label_ == "COLOR":
        color = ent.text
    elif ent.label_ == "PRICE":
        price = ent.text

# Print structured result
result = {
    "product": product or "null",
    "color": color or "null",
    "price": price or "null"
}
print(result)
