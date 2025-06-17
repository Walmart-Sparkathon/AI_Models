import spacy

nlp = spacy.load("./product_filter_nlp")

test_text = "Suggest black jeans under 2500"

doc = nlp(test_text)
print("Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
