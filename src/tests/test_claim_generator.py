from src.claims.claim_generator import Seq2SeqClaimGenerator


def test_chunked():
    data = list(range(10))
    # We use a small batch_size (e.g. 3) to test chunking
    chunks = list(Seq2SeqClaimGenerator._chunked(data, 3))
    assert len(chunks) == 4, "Should produce 4 chunks for range(10) with size=3"
    assert chunks[0] == [0, 1, 2]
    assert chunks[1] == [3, 4, 5]
    assert chunks[2] == [6, 7, 8]
    assert chunks[3] == [9]
