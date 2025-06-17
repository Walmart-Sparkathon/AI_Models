import spacy

nlp = spacy.load("./product_filter_nlp")

doc = nlp("Suggest black jeans under 2500")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Output:
# red (COLOR)
# saree (PRODUCT)
# 1500 (PRICE)
