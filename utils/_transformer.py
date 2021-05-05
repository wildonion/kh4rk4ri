







from sklearn.base import TransformerMixin


__all__ = ["Transformer", "ToDense"]

class Transformer(TransformerMixin):
    def transform(self, doc, **transform_params):
        return [self.clean_text(text) for text in doc]

    def fit(self, doc, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    def clean_text(self, text):
        return text.strip().lower()


class ToDense(TransformerMixin):
    def fit(self, doc, y=None, **fit_params):
        return self

    def transform(self, doc, y=None, **fit_params):
        return X.todense()