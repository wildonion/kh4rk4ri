







from sklearn.base import TransformerMixin


__all__ = ["Transformer"]


# NOTE - pipeline needs a class object that has fit and transform methods 
#        and because of that we're creating our cleaner transformer with
#        fit and transform methods.
class Transformer(TransformerMixin):
    def transform(self, doc, **transform_params):
        return [self.clean_text(text) for text in doc] # NOTE - doc is our dataset and text is a sample

    def fit(self, doc, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    def clean_text(self, text):
        return text.strip().lower()