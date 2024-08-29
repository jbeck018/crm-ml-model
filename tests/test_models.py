import unittest
import torch
from src.models import ChurnModel, HealthModel, ExpansionModel, UpsellModel

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = {
            'input_dim': 10,
            'hidden_dim': 20,
            'output_dim': 1,
            'learning_rate': 0.001,
            'num_upsell_categories': 5
        }
        torch.set_float32_matmul_precision('high')  # Enable TensorFloat-32 for better performance

    def test_churn_model(self):
        model = ChurnModel(self.config)
        model = torch.compile(model)
        x = torch.randn(1, 10)
        output = model(x)
        self.assertEqual(output.shape, (1, 1))

    def test_health_model(self):
        model = HealthModel(self.config)
        model = torch.compile(model)
        x = torch.randn(1, 10)
        output = model(x)
        self.assertEqual(output.shape, (1, 1))

    def test_expansion_model(self):
        model = ExpansionModel(self.config)
        model = torch.compile(model)
        x = torch.randn(1, 10)
        output = model(x)
        self.assertEqual(output.shape, (1, 1))

    def test_upsell_model(self):
        model = UpsellModel(self.config)
        model = torch.compile(model)
        x = torch.randn(1, 10)
        output = model(x)
        self.assertEqual(output.shape, (1, 5))

if __name__ == '__main__':
    unittest.main()