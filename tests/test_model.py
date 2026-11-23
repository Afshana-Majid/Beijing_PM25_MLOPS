import unittest
import joblib
import pandas as pd

class TestPM25Model(unittest.TestCase):

    def setUp(self):
        self.model = joblib.load("models/random_forest_model.pkl")
        self.data = pd.read_csv("data/processed/engineered_data.csv")

    def test_model_prediction(self):
        X = self.data.drop(columns=["PM2.5", "datetime"]).iloc[:5]

        predictions = self.model.predict(X)

        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(pd.notna(predictions)))


if __name__ == "__main__":
    unittest.main()