from unittest.mock import patch

from src.claims.device_selector import select_best_device


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.backends.mps.is_available", return_value=False)
def test_select_device_cuda(mock_mps, mock_cuda):
    device = select_best_device()
    assert device == "cuda", "Should select cuda when it is available."


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
def test_select_device_mps(mock_mps, mock_cuda):
    device = select_best_device()
    assert device == "mps", "Should select mps when cuda is not available but mps is."


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
def test_select_device_cpu(mock_mps, mock_cuda):
    device = select_best_device()
    assert device == "cpu", "Should select cpu when neither cuda nor mps is available."
