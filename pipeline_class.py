import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class SentimentPipeline:
    
    def __init__(self, model, vectorizer, label_encoder):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
    
    @staticmethod
    def normalize_negations(text):

        text = text.lower()

        text = re.sub(r"\b(dont|don't)\b", "do not", text)
        text = re.sub(r"\b(doesnt|doesn't)\b", "does not", text)
        text = re.sub(r"\b(didnt|didn't)\b", "did not", text)
        text = re.sub(r"\b(cant|can't)\b", "cannot", text)
        text = re.sub(r"\b(wont|won't)\b", "will not", text)
        text = re.sub(r"\b(isnt|isn't)\b", "is not", text)
        text = re.sub(r"\b(arent|aren't)\b", "are not", text)
        text = re.sub(r"\b(wasnt|wasn't)\b", "was not", text)
        text = re.sub(r"\b(werent|weren't)\b", "were not", text)
        text = re.sub(r"\b(couldnt|couldn't)\b", "could not", text)
        text = re.sub(r"\b(shouldnt|shouldn't)\b", "should not", text)
        text = re.sub(r"\b(wouldnt|wouldn't)\b", "would not", text)
        text = re.sub(r"\b(havent|haven't)\b", "have not", text)
        text = re.sub(r"\b(hasnt|hasn't)\b", "has not", text)
        text = re.sub(r"\b(hadnt|hadn't)\b", "had not", text)

        tokens = re.findall(r"[a-z]+", text)
        negators = {"not", "no", "never", "cannot"}
        out = []
        negate_next = False

        for tok in tokens:
            if tok in negators:
                negate_next = True
                out.append(tok)
                continue
            if negate_next:
                out.append(f"not_{tok}")
                negate_next = False
            else:
                out.append(tok)

        return " ".join(out)

    def _validate(self):
        if self.model is None or self.vectorizer is None or self.label_encoder is None:
            raise ValueError("model, vectorizer, and label_encoder must be set")

        checks = [
            ("vectorizer", self.vectorizer, "vocabulary_"),
            ("model", self.model, "classes_"),
            ("label_encoder", self.label_encoder, "classes_")
        ]
        
        for name, obj, attr in checks:
            if not hasattr(obj, attr):
                raise ValueError(f"{name} does not appear to be fitted")

    def _prepare_texts(self, texts, normalize=True):
        if texts is None:
            raise ValueError("texts cannot be None")

        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True
        elif not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a string or a list of strings")

        cleaned = []
        for text in texts:
            if text is None:
                cleaned.append("")
                continue
            text = str(text)
            cleaned.append(self.normalize_negations(text) if normalize else text)

        return cleaned, single

    def predict(self, texts, normalize=True, return_single=False):
        self._validate()
        cleaned, single = self._prepare_texts(texts, normalize)
        
        if len(cleaned) == 0:
            return "" if (return_single and single) else []

        X_tfidf = self.vectorizer.transform(cleaned)
        pred = self.model.predict(X_tfidf)
        labels = self.label_encoder.inverse_transform(pred)

        if return_single and single:
            return labels[0]
        return list(labels)

    def predict_proba(self, texts, normalize=True):
        self._validate()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("model does not support predict_proba")

        cleaned, _ = self._prepare_texts(texts, normalize)
        if len(cleaned) == 0:
            return []

        X_tfidf = self.vectorizer.transform(cleaned)
        return self.model.predict_proba(X_tfidf)
