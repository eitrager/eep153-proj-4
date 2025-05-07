
import unittest
import pandas as pd

class UnitTest(unittest.TestCase):

    # Testing to see if all the dataframes are float values
    def test_dataframe_all_floats(self):
        df = pd.DataFrame({
            'A': [1.1, 2.2, 3.3],
            'B': [4.4, 5.5, 6.6]
        })

        all_floats = df.stack().map(lambda x: isinstance(x, float)).all()
        self.assertTrue(all_floats, "Not all DataFrame elements are floats")

    # Testing to make sure to there are no NAs
    def test_no_nans(self):
        df = pd.DataFrame({
            'A': [1.1, 2.2, 3.3],
            'B': [4.4, 5.5, 6.6]
        })
        self.assertFalse(df.isnull().any().any(), "DataFrame contains NaN values")

    # Testing to see if all values of u in x series are kg
    def test_all_u_are_kg(self):
        # Simulating read_sheets function output, assuming it's a Series
        x = pd.Series(['kg', 'kg', 'mg', 'kg', 'g'], index=['u', 'u', 'c', 'u', 'z'])

        # Get all rows where index is 'u'
        values_at_u = x.loc['u']

        # Check that all of these values are 'kg'
        self.assertTrue((values_at_u == 'kg').all(), f"Not all 'u' values are 'kg':\n{values_at_u}")

if __name__ == "__main__":
    unittest.main()