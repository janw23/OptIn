from optin.utils import lerp, proximate


def test_lerp():
    a = 10.
    b = 100.
    assert proximate(lerp(0., a, b), a)
    assert proximate(lerp(1., a, b), b)
    assert proximate(lerp(0.5, a, b), 55.)
    assert proximate(lerp(0.22, a, b), 29.8)
