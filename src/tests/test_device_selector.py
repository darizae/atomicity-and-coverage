import pytest
from unittest.mock import patch

# Adjust import paths to match your project structure
from src.claims.device_selector import select_best_device, check_or_select_device


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


def test_check_or_select_device_no_request_auto(mocker):
    """
    If no device is requested, should auto-detect via select_best_device.
    We'll assume CPU is the fallback in a scenario with no cuda or mps.
    """
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    device = check_or_select_device(None)
    assert device == "cpu", "Should auto-select cpu when no device is requested."


def test_check_or_select_device_cpu():
    device = check_or_select_device("cpu")
    assert device == "cpu", "Should accept an explicit cpu request as valid."


@patch("torch.cuda.is_available", return_value=True)
def test_check_or_select_device_cuda_valid(mock_cuda):
    """
    If user explicitly requests cuda and it is available, we should use it.
    """
    device = check_or_select_device("cuda")
    assert device == "cuda", "Should accept an explicit cuda request if available."


@patch("torch.cuda.is_available", return_value=False)
def test_check_or_select_device_cuda_invalid(mock_cuda):
    """
    If user explicitly requests cuda and it is NOT available, raise a ValueError.
    """
    with pytest.raises(ValueError, match="CUDA not available"):
        check_or_select_device("cuda")


@patch("torch.backends.mps.is_available", return_value=True)
@patch("platform.system", return_value="Darwin")
def test_check_or_select_device_mps_valid(mock_platform, mock_mps):
    """
    If user explicitly requests mps and it is available, we should use it.
    """
    device = check_or_select_device("mps")
    assert device == "mps", "Should accept an explicit mps request if available."


@patch("torch.backends.mps.is_available", return_value=False)
@patch("platform.system", return_value="Darwin")
def test_check_or_select_device_mps_invalid(mock_platform, mock_mps):
    """
    If user explicitly requests mps and it is NOT available, raise a ValueError.
    """
    with pytest.raises(ValueError, match="MPS not available"):
        check_or_select_device("mps")


@patch("torch.cuda.is_available", return_value=True)
def test_check_or_select_device_cuda_colon_0(mock_cuda):
    """
    If user explicitly requests 'cuda:0' and CUDA is available, that should be valid.
    """
    device = check_or_select_device("cuda:0")
    assert device == "cuda:0", "Should accept an explicit cuda:0 request if CUDA is available."


def test_check_or_select_device_unknown():
    """
    If user explicitly requests something we don't recognize, raise an error.
    """
    with pytest.raises(ValueError, match="Unknown or unsupported device requested"):
        check_or_select_device("some_unknown_device")
