from router.features.extractor import extract


def test_short_qa_is_lightweight():
    f = extract("what is the capital of France")
    assert f.tool_required is False
    assert f.code_fence_count == 0
    assert f.has_path_ref is False
    assert f.has_url is False
    assert f.sensitive is False
    assert f.qa_keywords >= 1


def test_path_ref_and_agent_keyword_imply_tool_required():
    f = extract("refactor src/auth/login.py to use async/await")
    assert f.has_path_ref is True
    assert f.agent_keywords >= 1
    assert f.tool_required is True


def test_code_fence_counted_and_ratio():
    prompt = "look at this:\n```py\ndef x():\n    return 1\n```\n"
    f = extract(prompt)
    assert f.code_fence_count == 1
    assert 0 < f.code_ratio <= 1


def test_url_detection():
    f = extract("summarize https://example.com")
    assert f.has_url is True


def test_secret_detection_aws():
    f = extract("here is my key: AKIAABCDEFGHIJKLMNOP and use it")
    assert f.sensitive is True


def test_secret_detection_pem():
    f = extract("-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----")
    assert f.sensitive is True


def test_token_estimate_floor():
    f = extract("hi")
    assert f.n_tokens_est >= 1
