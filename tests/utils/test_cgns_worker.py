# %% Imports


# %% Tests
class Test_cgns_worker:
    def test_import(self):
        from plaid.utils import cgns_worker

        cgns_worker.logger
