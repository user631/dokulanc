from __future__ import annotations

import random
import re
from datetime import date

COUNTY_CODES = [f"{i:02d}" for i in list(range(2, 21)) + list(range(22, 45)) + [51]]
_EPOCH = date(1867, 1, 1)


def digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _weighted(ds: str, weights: list[int]) -> int:
    return sum(int(d) * weights[i % len(weights)] for i, d in enumerate(ds))


def tax_number_check_digit(first7: str) -> int:
    s = _weighted(first7, [9, 7, 3, 1, 9, 7, 3])
    return (10 - s % 10) % 10


def is_valid_tax_number(value: str) -> bool:
    if not re.fullmatch(r"\s*\d{8}-?\d-?\d{2}\s*", value or ""):
        return False
    d = digits(value)
    if tax_number_check_digit(d[:7]) != int(d[7]):
        return False
    if d[8] not in "12345":
        return False
    return d[9:11] in COUNTY_CODES


def generate_tax_number(rng: random.Random) -> str:
    first7 = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(6))
    base = first7 + str(tax_number_check_digit(first7))
    return f"{base}-{rng.choice('2222345')}-{rng.choice(COUNTY_CODES)}"


def is_valid_taj(value: str) -> bool:
    d = digits(value)
    if len(d) != 9:
        return False
    return _weighted(d[:8], [3, 7]) % 10 == int(d[8])


def generate_taj(rng: random.Random) -> str:
    first8 = "".join(str(rng.randint(0, 9)) for _ in range(8))
    d = first8 + str(_weighted(first8, [3, 7]) % 10)
    return f"{d[:3]} {d[3:6]} {d[6:]}"


def tax_id_birth_date(value: str) -> date | None:
    d = digits(value)
    if len(d) != 10:
        return None
    return date.fromordinal(_EPOCH.toordinal() + int(d[1:6]))


def is_valid_tax_id(value: str) -> bool:
    d = digits(value)
    if len(d) != 10 or d[0] != "8":
        return False
    check = sum(int(x) * (i + 1) for i, x in enumerate(d[:9])) % 11
    return check != 10 and check == int(d[9])


def generate_tax_id(rng: random.Random, birth: date) -> str:
    days = (birth - _EPOCH).days
    while True:
        first9 = "8" + f"{days:05d}" + "".join(str(rng.randint(0, 9)) for _ in range(3))
        check = sum(int(x) * (i + 1) for i, x in enumerate(first9)) % 11
        if check != 10:
            return first9 + str(check)


def is_valid_hu_account(value: str) -> bool:
    d = digits(value)
    if len(d) not in (16, 24):
        return False
    w = [9, 7, 3, 1]
    return _weighted(d[:8], w) % 10 == 0 and _weighted(d[8:], w) % 10 == 0


def _account_block(rng: random.Random, length: int) -> str:
    body = "".join(str(rng.randint(0, 9)) for _ in range(length - 1))
    s = _weighted(body, [9, 7, 3, 1])
    w_last = [9, 7, 3, 1][(length - 1) % 4]
    for c in range(10):
        if (s + c * w_last) % 10 == 0:
            return body + str(c)
    raise ValueError("no check digit")


def generate_hu_account(rng: random.Random, length: int = 24) -> str:
    d = _account_block(rng, 8) + _account_block(rng, length - 8)
    return "-".join(d[i:i + 8] for i in range(0, len(d), 8))


def _iban_numeric(s: str) -> int:
    return int("".join(str(int(ch, 36)) for ch in s))


def is_valid_iban(value: str) -> bool:
    s = re.sub(r"\s", "", value or "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{8,30}", s):
        return False
    if s.startswith("HU") and len(s) != 28:
        return False
    return _iban_numeric(s[4:] + s[:4]) % 97 == 1


def iban_from_hu_account(account: str) -> str:
    acct = digits(account).ljust(24, "0")
    check = 98 - _iban_numeric(acct + "HU00") % 97
    raw = f"HU{check:02d}{acct}"
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def hu_account_from_iban(iban: str) -> str | None:
    s = re.sub(r"\s", "", iban or "").upper()
    if not s.startswith("HU") or len(s) != 28:
        return None
    return s[4:]
