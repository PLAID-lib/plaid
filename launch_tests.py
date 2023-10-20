import pytest

def run_tests(test_path):
    pytest.main([test_path])

if __name__ == '__main__':
    run_tests('tests/')