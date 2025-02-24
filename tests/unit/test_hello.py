from blueprint.hello import hello


def test_hello():
    assert hello() == "Hello, world!"
