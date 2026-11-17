import random
from datetime import date

import pytest

from dokulanc import identifiers as ids


@pytest.fixture
def rng():
    return random.Random(123)


def test_generated_tax_numbers_are_valid(rng):
    for _ in range(200):
        assert ids.is_valid_tax_number(ids.generate_tax_number(rng))


@pytest.mark.parametrize("value", ["12345678-1-42", "1234567-1-42", "abc", "", "12345670-9-42", "12345670-2-21"])
def test_invalid_tax_numbers(value):
    assert not ids.is_valid_tax_number(value)


def test_tax_number_check_digit_changes_validity(rng):
    t = ids.generate_tax_number(rng)
    d = ids.digits(t)
    wrong = f"{d[:7]}{(int(d[7]) + 1) % 10}-{d[8]}-{d[9:]}"
    assert not ids.is_valid_tax_number(wrong)


def test_known_iban_example():
    assert ids.is_valid_iban("HU42 1177 3016 1111 1018 0000 0000")
    assert ids.is_valid_hu_account("11773016-11111018")
    assert not ids.is_valid_iban("HU43 1177 3016 1111 1018 0000 0000")


def test_generated_accounts_and_ibans(rng):
    for length in (16, 24):
        for _ in range(100):
            acct = ids.generate_hu_account(rng, length)
            assert ids.is_valid_hu_account(acct)
            iban = ids.iban_from_hu_account(acct)
            assert ids.is_valid_iban(iban)
            assert ids.hu_account_from_iban(iban).startswith(ids.digits(acct))


def test_taj(rng):
    for _ in range(100):
        assert ids.is_valid_taj(ids.generate_taj(rng))
    assert not ids.is_valid_taj("123 456 789")


def test_tax_id_encodes_birth_date(rng):
    birth = date(1999, 4, 12)
    tid = ids.generate_tax_id(rng, birth)
    assert ids.is_valid_tax_id(tid)
    assert ids.tax_id_birth_date(tid) == birth
    assert not ids.is_valid_tax_id(tid[:9] + str((int(tid[9]) + 1) % 10))
