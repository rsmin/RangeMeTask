from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import general_functions as gf
from sklearn.metrics import accuracy_score
import pandas as pd
import eli5
import warnings

warnings.filterwarnings('ignore')

_PARAMS = gf.params_loader()
_FEATURES = _PARAMS['category_features'] + _PARAMS['numeric_features']
_TARGET = _PARAMS['target']
_DF = gf.data_loader()


class PrePipeline:
    def __init__(self):
        self.category_pipeline = Pipeline(
        steps=[
            ('cat_selector', gf.FeatureSelector(_PARAMS['category_features'])),
            ('cat_formatter', gf.CategoryFormatter(_PARAMS['invalid_values'])),
            ('cat_imputer', gf.Imputer(strategy='most_frequent')),
            ('cat_encoder', gf.Encoder()),
            ('cat_filter', gf.FeatureFilter())
        ]
    )
        self.numerical_pipeline = Pipeline(
        steps=[
            ('numeric_selector',
             gf.FeatureSelector(_PARAMS['numeric_features'])),
            ('numeric_formatter', gf.NumericFormatter()),
            ('numeric_imputer', gf.Imputer(strategy='median')),
            ('numeric_filter', gf.FeatureFilter())
        ]
    )

    def fit(self, x):
        self.category_pipeline.fit(x)
        self.numerical_pipeline.fit(x)
        return self

    def transform(self, x):
        x_cat = self.category_pipeline.transform(x)
        x_num = self.numerical_pipeline.transform(x)
        x_union = pd.concat([x_cat, x_num], axis=1)
        return x_union

    def fit_transform(self, x):
        self.fit(x)
        x_union = self.transform(x)
        return x_union


class Modeling():
    def __init__(self):
        self.model = XGBClassifier(**_PARAMS['model_params'])
        self.pre_pipe = PrePipeline()
        x_train_raw, x_test_raw, self.y_train, self.y_test = gf.data_split(
            input_df=_DF,
            feature_cols=_FEATURES,
            target_col=_TARGET)

        self.x_train = self.pre_pipe.fit_transform(x_train_raw)
        self.x_test = self.pre_pipe.transform(x_test_raw)

    def train(self):
        self.model.fit(self.x_train, self.y_train,
                       early_stopping_rounds=10,
                       eval_set=[(self.x_test, self.y_test)],verbose=False)

    def pred(self, x=None):
        if x is None:
            x = self.x_test
        y_pred = self.model.predict(x)
        return y_pred

    def accuracy(self):
        return accuracy_score(self.y_test, self.pred(), normalize = True)

    def feature_weights(self):
        exp_weights = eli5.explain_weights(self.model)
        feature_importance = exp_weights.feature_importances.importances
        weights_dict = {fi.feature:fi.weight for fi in feature_importance}
        return weights_dict


if __name__ == '__main__':
    m1 = Modeling()
    m1.train()
    print('model accuracy: ', m1.accuracy())
    print('feature importance: \n', m1.feature_weights())

    # save predict upgrade company ids
    X = _DF[_FEATURES]
    Y = _DF[_TARGET]
    X_fit = m1.pre_pipe.transform(X)
    Y_pred = m1.pred(X_fit)
    promote_suppliers = _DF[Y_pred==1]['company_id']
    promote_suppliers.to_csv('Data/updated_company_id_predict.txt', index=False)

