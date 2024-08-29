import pandas as pd
from typing import Dict, Any

class CRMDataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        # Load data from various sources (CRM, market data, usage data, etc.)
        # This is a placeholder implementation
        crm_data = pd.read_csv(self.config['crm_data_path'])
        market_data = pd.read_csv(self.config['market_data_path'])
        usage_data = pd.read_csv(self.config['usage_data_path'])

        # Merge data based on specific IDs
        merged_data = pd.merge(crm_data, market_data, on='account_id', how='left')
        merged_data = pd.merge(merged_data, usage_data, on='account_id', how='left')

        return merged_data