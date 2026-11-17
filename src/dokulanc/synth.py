from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import identifiers as ids
from .normalize import format_amount, format_date, strip_accents

FONT_DIR = Path("C:/Windows/Fonts")
SANS = [("arial.ttf", "arialbd.ttf"), ("calibri.ttf", "calibrib.ttf"), ("verdana.ttf", "verdanab.ttf"),
        ("tahoma.ttf", "tahomabd.ttf"), ("segoeui.ttf", "segoeuib.ttf"), ("trebuc.ttf", "trebucbd.ttf")]
SERIF = [("times.ttf", "timesbd.ttf"), ("georgia.ttf", "georgiab.ttf")]
MONO = [("consola.ttf", "consolab.ttf"), ("cour.ttf", "courbd.ttf"), ("lucon.ttf", "lucon.ttf")]
HAND = ["segoepr.ttf", "Inkfree.ttf", "LHANDW.TTF", "segoesc.ttf", "comic.ttf"]

A4 = (1240, 1754)

PRODUCTS = [
    ("Szoftverfejlesztési szolgáltatás", "óra", 9000, 18000, 27),
    ("Rendszerüzemeltetés havidíj", "hó", 40000, 180000, 27),
    ("Tanácsadás", "óra", 12000, 30000, 27),
    ("Irodaszer csomag", "db", 1500, 9000, 27),
    ("Nyomtatópapír A4 500 lap", "csomag", 1800, 3200, 27),
    ("Laptop táska", "db", 6000, 18000, 27),
    ("USB-C töltő 65W", "db", 7000, 16000, 27),
    ("Monitor 27\"", "db", 55000, 140000, 27),
    ("Könyv: Gépi tanulás alapjai", "db", 4500, 12000, 5),
    ("Szakmai folyóirat előfizetés", "év", 9000, 30000, 5),
    ("Kenyér 1 kg", "db", 600, 1200, 5),
    ("Ásványvíz 1,5 l", "karton", 1200, 2600, 18),
    ("Tejtermék csomag", "csomag", 2500, 6000, 18),
    ("Szállítási díj", "alkalom", 1500, 5000, 27),
    ("Takarítási szolgáltatás", "alkalom", 15000, 45000, 27),
    ("Webtárhely bérlés", "év", 12000, 60000, 27),
    ("Domain regisztráció", "db", 3000, 8000, 27),
    ("Fordítás angolról magyarra", "oldal", 3500, 7000, 27),
    ("Rendezvényszervezés", "alkalom", 80000, 300000, 27),
    ("Gépjármű szerviz munkadíj", "óra", 9000, 16000, 27),
]

RECEIPT_ITEMS = [
    ("TEJ 2,8% 1L", 329, 459), ("KENYER FEHER 1KG", 549, 899), ("VAJ 200G", 899, 1299),
    ("TOJAS 10DB", 999, 1599), ("ALMA 1KG", 499, 799), ("BANAN 1KG", 599, 849),
    ("SAJT TRAPPISTA 500G", 1899, 2799), ("JOGHURT NATUR", 199, 349), ("KAVE 250G", 1799, 2999),
    ("CUKOR 1KG", 399, 599), ("LISZT BL55 1KG", 299, 449), ("ASVANYVIZ 1,5L", 149, 259),
    ("PAPIRZSEBKENDO", 299, 599), ("MOSOGATOSZER 1L", 699, 1199), ("CSOKOLADE 100G", 399, 799),
    ("PARIZSI 300G", 799, 1299), ("PAPRIKA KG", 999, 1899), ("PARADICSOM KG", 899, 1599),
    ("UDITO 2L", 499, 899), ("ZSEMLE", 59, 99), ("TESZTA 500G", 349, 699), ("RIZS 1KG", 599, 999),
]

LETTER_SCENARIOS = [
    {
        "targy": "Fizetési felszólítás",
        "teendo": "a {szamla} számú számla {osszeg} összegű tartozásának kiegyenlítése",
        "body": [
            "Nyilvántartásunk szerint a(z) {szamla} számú, {kelt} kelt számla {osszeg} összegű ellenértéke a mai napig nem érkezett meg számlánkra.",
            "Kérjük, hogy a fenti összeget legkésőbb {hatarido} napjáig szíveskedjen átutalni a {bank} számú bankszámlánkra.",
            "Amennyiben a befizetést időközben teljesítette, levelünket tekintse tárgytalannak.",
        ],
    },
    {
        "targy": "Hiánypótlási felhívás",
        "teendo": "a hiányzó dokumentumok ({dok}) benyújtása",
        "body": [
            "A(z) {ugyszam} ügyszámú kérelmének vizsgálata során megállapítottuk, hogy az alábbi dokumentum hiányzik: {dok}.",
            "Felhívjuk, hogy a hiányzó dokumentumot legkésőbb {hatarido} napjáig nyújtsa be ügyfélszolgálatunkon vagy postai úton.",
            "A határidő elmulasztása esetén kérelmét a rendelkezésre álló adatok alapján bíráljuk el.",
        ],
    },
    {
        "targy": "Szerződésmódosítás",
        "teendo": "a módosított szerződés aláírása és visszaküldése",
        "body": [
            "Tájékoztatjuk, hogy a közöttünk {kelt} napján létrejött szolgáltatási szerződés díjtételei a jövő hónaptól módosulnak.",
            "A módosított szerződést mellékelten megküldjük. Kérjük, hogy azt aláírva legkésőbb {hatarido} napjáig juttassa vissza részünkre.",
            "Kérdés esetén ügyfélszolgálatunk munkanapokon 8 és 16 óra között készséggel áll rendelkezésére.",
        ],
    },
    {
        "targy": "Meghívó szakmai egyeztetésre",
        "teendo": "részvétel visszaigazolása",
        "body": [
            "Ezúton meghívjuk a(z) {ugyszam} azonosítójú projekt soron következő szakmai egyeztetésére.",
            "Az egyeztetés helyszíne irodánk tárgyalója, időpontja {hatarido} 10 óra.",
            "Kérjük, részvételi szándékát válaszlevélben jelezze felénk.",
        ],
    },
]

DOCS_MISSING = ["jövedelemigazolás", "lakcímkártya másolata", "aláírt meghatalmazás", "orvosi igazolás",
                "munkáltatói igazolás", "bankszámlakivonat"]
SCHOOLS = [
    ("Budapesti Műszaki és Gazdaságtudományi Egyetem", ["mérnökinformatikus BSc", "villamosmérnök BSc", "gépészmérnök MSc"]),
    ("Óbudai Egyetem", ["mérnökinformatikus BSc", "gazdaságinformatikus BSc", "mechatronikai mérnök BSc"]),
    ("Eötvös Loránd Tudományegyetem", ["programtervező informatikus BSc", "matematika BSc", "pszichológia MA"]),
    ("Debreceni Egyetem", ["közgazdász BSc", "gazdaságinformatikus MSc"]),
    ("Szegedi Tudományegyetem", ["biológus BSc", "programtervező informatikus MSc"]),
    ("Pécsi Tudományegyetem", ["jogász", "anglisztika BA"]),
    ("Corvinus Egyetem", ["gazdálkodás és menedzsment BSc", "pénzügy MSc"]),
]
HIGH_SCHOOLS = ["Petőfi Sándor Gimnázium", "Bolyai János Gimnázium", "Kossuth Lajos Szakgimnázium",
                "Széchenyi István Technikum", "Arany János Gimnázium"]
POSITIONS = ["szoftverfejlesztő", "rendszergazda", "adatelemző", "projektmenedzser", "ügyfélszolgálati munkatárs",
             "pénzügyi asszisztens", "értékesítő", "tesztmérnök", "junior fejlesztő", "irodavezető"]
LANGUAGES = ["angol (C1)", "angol (B2)", "német (B1)", "német (B2)", "francia (A2)", "spanyol (B1)", "olasz (A2)"]
SKILLS = ["Python", "Java", "SQL", "Excel", "Git", "Docker", "projektmenedzsment", "csapatmunka", "prezentáció",
          "B kategóriás jogosítvány", "Linux", "C#", "JavaScript", "ügyfélkezelés", "SAP"]
FORM_TITLES = ["IGÉNYLŐLAP", "ADATLAP", "JELENTKEZÉSI LAP", "REGISZTRÁCIÓS ŰRLAP", "ADATMÓDOSÍTÓ NYOMTATVÁNY"]
PAYMENT = ["átutalás", "készpénz", "bankkártya"]


@dataclass
class Rendered:
    image: Image.Image
    doc_type: str
    fields: dict
    lines: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class Canvas:
    def __init__(self, size, rng: random.Random, background=None):
        self.rng = rng
        shade = rng.randint(242, 255)
        self.img = Image.new("RGB", size, background or (shade, shade, max(shade - rng.randint(0, 6), 230)))
        self.draw = ImageDraw.Draw(self.img)
        self.lines: list[dict] = []
        self._fonts: dict = {}

    def font(self, name: str, size: int) -> ImageFont.FreeTypeFont:
        key = (name, size)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(str(FONT_DIR / name), size)
        return self._fonts[key]

    def text(self, xy, text: str, font: str, size: int, fill=(20, 20, 20), anchor="la", record=True):
        if not text:
            return (xy[0], xy[1], xy[0], xy[1])
        f = self.font(font, size)
        self.draw.text(xy, text, font=f, fill=fill, anchor=anchor)
        box = self.draw.textbbox(xy, text, font=f, anchor=anchor)
        if record:
            self.lines.append({"text": text, "bbox": [int(v) for v in box]})
        return box

    def width(self, text: str, font: str, size: int) -> int:
        b = self.draw.textbbox((0, 0), text, font=self.font(font, size))
        return b[2] - b[0]

    def wrap(self, text: str, font: str, size: int, max_width: int) -> list[str]:
        words, lines, cur = text.split(), [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if self.width(cand, font, size) <= max_width:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def rect(self, box, width=2, outline=(40, 40, 40), fill=None):
        self.draw.rectangle(box, outline=outline, width=width, fill=fill)

    def hline(self, x1, x2, y, width=2, fill=(60, 60, 60)):
        self.draw.line([(x1, y), (x2, y)], fill=fill, width=width)


class Generator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.fake = Faker("hu_HU")
        self.fake.seed_instance(seed)
        self.np_rng = np.random.default_rng(seed)

    def choice(self, seq):
        return self.rng.choice(seq)

    def random_date(self, start=date(2024, 1, 1), end=date(2026, 12, 31)) -> date:
        return start + timedelta(days=self.rng.randint(0, (end - start).days))

    def company(self) -> str:
        name = self.fake.last_name() + self.choice([" és Társa", " Trade", " Szolgáltató", " Consulting", " Tech", " Logisztika", ""])
        return f"{name} {self.choice(['Kft.', 'Bt.', 'Zrt.', 'Kft.'])}"

    def address(self) -> str:
        return f"{self.fake.postcode()} {self.fake.city()}, {self.fake.street_address()}"

    def phone(self) -> str:
        return self.choice(["+36 30 ", "+36 20 ", "+36 70 ", "06 1 "]) + f"{self.rng.randint(100, 999)} {self.rng.randint(1000, 9999)}"

    def email(self, name: str) -> str:
        parts = strip_accents(name.lower()).replace("dr. ", "").split()
        user = ".".join(parts[-2:]) if len(parts) >= 2 else parts[0]
        return f"{user}@{self.choice(['gmail.com', 'freemail.hu', 'citromail.hu', 'outlook.com', 'example.hu'])}"

    def person(self) -> str:
        return f"{self.fake.last_name()} {self.fake.first_name()}"

    def fonts(self, serif_ok=True):
        pool = SANS + (SERIF if serif_ok else [])
        return self.choice(pool)

    def generate(self, doc_type: str, corrupt: bool = False) -> Rendered:
        return getattr(self, f"gen_{doc_type}")(corrupt)

    def gen_szamla(self, corrupt: bool = False) -> Rendered:
        c = Canvas(A4, self.rng)
        reg, bold = self.fonts()
        s = self.rng.randint(22, 27)
        m = self.rng.randint(70, 110)
        W = A4[0] - m
        issue = self.random_date()
        perform = issue - timedelta(days=self.rng.randint(0, 5))
        due = issue + timedelta(days=self.choice([8, 8, 15, 15, 30]))
        dstyle = self.rng.randint(0, 3)
        prefix = self.choice(["SZ", "INV", "E", "KFT", "A"])
        number = f"{prefix}-{issue.year}/{self.rng.randint(1, 9999):05d}"
        seller, buyer = self.company(), self.company() if self.rng.random() < 0.8 else self.person()
        seller_addr, buyer_addr = self.address(), self.address()
        seller_tax = ids.generate_tax_number(self.rng)
        buyer_tax = ids.generate_tax_number(self.rng) if buyer.endswith(("Kft.", "Bt.", "Zrt.")) else ""
        acct = ids.generate_hu_account(self.rng, self.choice([16, 24]))
        bank = ids.iban_from_hu_account(acct) if self.rng.random() < 0.4 else acct
        pay = self.choice(["átutalás", "átutalás", "bankkártya", "készpénz"])
        decimals = 0
        errors = []
        kind = self.choice(["osszeg", "adoszam", "hatarido"]) if corrupt else None
        if kind == "adoszam":
            d = ids.digits(seller_tax)
            seller_tax = f"{d[:7]}{(int(d[7]) + self.rng.randint(1, 9)) % 10}-{d[8]}-{d[9:]}"
            errors.append("elado_adoszam")
        elif kind == "hatarido":
            due = issue - timedelta(days=self.rng.randint(3, 20))
            errors.append("fizetesi_hatarido")

        title = self.choice(["SZÁMLA", "Számla", "SZÁMLA / INVOICE", "Elektronikus számla"])
        y = m
        if self.rng.random() < 0.5:
            c.rect((m, y, m + 160, y + 90), width=3, outline=(90, 90, 160))
            c.text((m + 80, y + 45), seller.split()[0].upper()[:10], bold, 24, fill=(90, 90, 160), anchor="mm")
            c.text((W, y + 10), title, bold, s + 18, anchor="ra")
            c.text((W, y + 70), f"Sorszám: {number}", reg, s, anchor="ra")
            y += 130
        else:
            c.text((A4[0] // 2, y), title, bold, s + 18, anchor="ma")
            c.text((A4[0] // 2, y + 60), f"Számla sorszáma: {number}", reg, s, anchor="ma")
            y += 120
        c.hline(m, W, y)
        y += 20

        col2 = A4[0] // 2 + 10
        swap = self.rng.random() < 0.3
        blocks = [("Eladó", seller, seller_addr, seller_tax), ("Vevő", buyer, buyer_addr, buyer_tax)]
        if swap:
            blocks.reverse()
        yy = y
        for (label, name, addr, tax), x in zip(blocks, [m, col2]):
            yl = y
            c.text((x, yl), f"{label}:", bold, s)
            yl += s + 12
            c.text((x, yl), name, bold, s)
            yl += s + 10
            for part in c.wrap(addr, reg, s - 2, col2 - m - 30):
                c.text((x, yl), part, reg, s - 2)
                yl += s + 6
            if tax:
                c.text((x, yl), f"Adószám: {tax}", reg, s - 2)
                yl += s + 6
            yy = max(yy, yl)
        y = yy + 20
        c.hline(m, W, y, width=1)
        y += 18
        date_items = [("Kiállítás dátuma", format_date(issue, dstyle)), ("Teljesítés dátuma", format_date(perform, dstyle)),
                      ("Fizetési határidő", format_date(due, dstyle)), ("Fizetési mód", pay)]
        if self.rng.random() < 0.5:
            cw = (W - m) // 4
            for i, (k, v) in enumerate(date_items):
                c.text((m + i * cw, y), k, bold, s - 5)
                c.text((m + i * cw, y + s + 4), v, reg, s - 3)
            y += 2 * s + 30
        else:
            for k, v in date_items:
                c.text((m, y), f"{k}:", bold, s - 2)
                c.text((m + 300, y), v, reg, s - 2)
                y += s + 8
            y += 12

        n_items = self.rng.randint(1, 7)
        items = []
        for p in self.rng.sample(PRODUCTS, n_items):
            name, unit, lo, hi, rate = p
            qty = self.rng.choice([1, 1, 1, 2, 3, 4, 5, 10, 12])
            price = round(self.rng.randint(lo, hi), -1)
            net = qty * price
            vat = round(net * rate / 100)
            items.append({"megnevezes": name, "mennyiseg": f"{qty} {unit}", "egysegar": format_amount(price, decimals, ""),
                          "netto": format_amount(net, decimals, ""), "afa_kulcs": f"{rate}%",
                          "afa": format_amount(vat, decimals, ""), "brutto": format_amount(net + vat, decimals, ""),
                          "_net": net, "_vat": vat})
        cols = [("Megnevezés", 0.0, "la"), ("Menny.", 0.40, "la"), ("Egységár", 0.62, "ra"), ("Nettó", 0.74, "ra"),
                ("ÁFA%", 0.80, "ra"), ("ÁFA", 0.895, "ra"), ("Bruttó", 1.0, "ra")]
        tw = W - m
        ts = s - 5
        c.rect((m - 6, y - 6, W + 6, y + ts + 14), width=1, outline=(80, 80, 80), fill=(225, 225, 225) if self.rng.random() < 0.6 else None)
        for label, pos, anchor in cols:
            c.text((m + int(pos * tw), y), label, bold, ts, anchor=anchor)
        y += ts + 22
        for it in items:
            name_lines = c.wrap(it["megnevezes"], reg, ts, int(0.38 * tw))
            vals = [it["mennyiseg"], it["egysegar"], it["netto"], it["afa_kulcs"], it["afa"], it["brutto"]]
            for (label, pos, anchor), v in zip(cols[1:], vals):
                c.text((m + int(pos * tw), y), v, reg, ts, anchor=anchor)
            for i, nl in enumerate(name_lines):
                c.text((m, y + i * (ts + 6)), nl, reg, ts)
            y += len(name_lines) * (ts + 6) + 10
            c.hline(m, W, y - 4, width=1, fill=(170, 170, 170))
        net_total = sum(i["_net"] for i in items)
        vat_total = sum(i["_vat"] for i in items)
        gross = net_total + vat_total
        printed_gross = gross
        if kind == "osszeg":
            printed_gross = gross + self.choice([-1000, 100, 900, 10000, -50])
            errors.append("brutto_osszesen")
        y += 20
        totals = [("Nettó összesen:", format_amount(net_total, decimals)), ("ÁFA összesen:", format_amount(vat_total, decimals)),
                  (self.choice(["Bruttó végösszeg:", "Fizetendő összesen:", "Végösszeg:"]), format_amount(printed_gross, decimals))]
        for i, (k, v) in enumerate(totals):
            f = bold if i == 2 else reg
            c.text((W - 330, y), k, f, s - 1, anchor="ra")
            c.text((W, y), v, f, s - 1, anchor="ra")
            y += s + 12
        y += 30
        c.text((m, y), f"Bankszámlaszám: {bank}", reg, s - 3)
        y += s + 6
        if self.rng.random() < 0.5:
            c.text((m, y), "Kérjük, átutaláskor a közleményben tüntesse fel a számla sorszámát.", reg, s - 6, fill=(80, 80, 80))
        c.text((A4[0] // 2, A4[1] - m), self.choice(["Köszönjük, hogy minket választott!", "A számla a 23/2014. NGM rendelet alapján készült.", ""]),
               reg, s - 7, anchor="md", fill=(110, 110, 110))

        fields = {
            "szamlaszam": number, "kiallitas_datuma": format_date(issue, dstyle),
            "teljesites_datuma": format_date(perform, dstyle), "fizetesi_hatarido": format_date(due, dstyle),
            "elado_nev": seller, "elado_cim": seller_addr, "elado_adoszam": seller_tax,
            "vevo_nev": buyer, "vevo_cim": buyer_addr, "vevo_adoszam": buyer_tax or None,
            "bankszamla": bank, "fizetesi_mod": pay,
            "netto_osszesen": format_amount(net_total, decimals), "afa_osszesen": format_amount(vat_total, decimals),
            "brutto_osszesen": format_amount(printed_gross, decimals),
            "tetelek": [{k: v for k, v in it.items() if not k.startswith("_")} for it in items],
        }
        return Rendered(c.img, "szamla", fields, c.lines, errors)

    def gen_nyugta(self, corrupt: bool = False) -> Rendered:
        reg, bold = self.choice(MONO)
        s = self.rng.randint(20, 24)
        width = self.rng.randint(520, 640)
        items = []
        for name, lo, hi in self.rng.sample(RECEIPT_ITEMS, self.rng.randint(2, 12)):
            qty = self.choice([1, 1, 1, 1, 2, 3])
            price = self.rng.randint(lo, hi)
            items.append((name, qty, price))
        height = 520 + len(items) * (s + 8) * 1.4 + 300
        c = Canvas((width, int(height)), self.rng)
        m = 24
        W = width - m
        y = 30
        shop = f"{self.fake.last_name()} {self.choice(['ABC', 'Kisbolt', 'Élelmiszer', 'Market', 'Csemege', 'Diszkont'])} {self.choice(['Kft.', 'Bt.'])}"
        addr = self.address()
        tax = ids.generate_tax_number(self.rng)
        when = self.random_date()
        hour = f"{self.rng.randint(7, 21):02d}:{self.rng.randint(0, 59):02d}"
        number = f"{self.rng.randint(1, 9999):04d}/{self.rng.randint(1, 99999):05d}"
        pay = self.choice(["KÉSZPÉNZ", "BANKKÁRTYA", "BANKKÁRTYA", "SZÉP KÁRTYA"])
        c.text((width // 2, y), shop.upper(), bold, s + 2, anchor="ma")
        y += s + 12
        for part in c.wrap(addr, reg, s - 2, W - m):
            c.text((width // 2, y), part, reg, s - 2, anchor="ma")
            y += s + 4
        c.text((width // 2, y), f"ADÓSZÁM: {tax}", reg, s - 2, anchor="ma")
        y += s + 18
        c.text((width // 2, y), "NYUGTA", bold, s + 4, anchor="ma")
        y += s + 22
        total = 0
        lines_items = []
        for name, qty, price in items:
            amount = qty * price
            total += amount
            if qty > 1:
                c.text((m, y), f"{qty} DB x {price}", reg, s - 3)
                y += s + 2
            c.text((m, y), name, reg, s)
            c.text((W, y), f"{amount} {self.choice(['C00', 'B00', 'A00'])}", reg, s, anchor="ra")
            y += s + 8
            lines_items.append({"megnevezes": name, "osszeg": str(amount)})
        c.hline(m, W, y, width=1)
        y += 12
        printed_total = total
        errors = []
        if corrupt:
            printed_total = total + self.choice([-100, 10, 500, 1000])
            errors.append("vegosszeg")
        label = self.choice(["ÖSSZESEN:", "FIZETENDŐ:", "VÉGÖSSZEG:"])
        c.text((m, y), label, bold, s + 4)
        c.text((W, y), format_amount(printed_total, 0), bold, s + 4, anchor="ra")
        y += s + 18
        c.text((m, y), pay, reg, s)
        c.text((W, y), format_amount(printed_total, 0), reg, s, anchor="ra")
        y += s + 22
        date_s = f"{when.year}.{when.month:02d}.{when.day:02d}."
        c.text((m, y), date_s, reg, s - 2)
        c.text((W, y), hour, reg, s - 2, anchor="ra")
        y += s + 8
        c.text((m, y), f"NYUGTASZÁM: {number}", reg, s - 2)
        y += s + 8
        c.text((m, y), f"AP A{self.rng.randint(10000000, 99999999)}", reg, s - 2)
        y += s + 20
        c.text((width // 2, y), self.choice(["KÖSZÖNJÜK A VÁSÁRLÁST!", "VISZONTLÁTÁSRA!", "KÖSZÖNJÜK!"]), bold, s - 1, anchor="ma")
        y += s + 40
        c.img = c.img.crop((0, 0, width, min(int(height), y)))
        fields = {
            "uzlet_nev": shop.upper(), "uzlet_cim": addr, "adoszam": tax, "datum": date_s, "ido": hour,
            "nyugtaszam": number, "fizetesi_mod": pay, "vegosszeg": format_amount(printed_total, 0),
            "tetelek": lines_items,
        }
        return Rendered(c.img, "nyugta", fields, c.lines, errors)

    def gen_level(self, corrupt: bool = False) -> Rendered:
        c = Canvas(A4, self.rng)
        reg, bold = self.fonts()
        s = self.rng.randint(23, 27)
        m = self.rng.randint(100, 140)
        W = A4[0] - m
        sender = self.company()
        sender_addr = self.address()
        recipient = ("Dr. " if self.rng.random() < 0.15 else "") + self.person()
        recipient_addr = self.address()
        when = self.random_date()
        due = when + timedelta(days=self.rng.randint(8, 30))
        dstyle = self.choice([0, 3, 3])
        sc = self.choice(LETTER_SCENARIOS)
        ref = f"{self.choice(['ÜSZ', 'IKT', 'REF', 'Ü'])}-{when.year}/{self.rng.randint(100, 99999)}"
        vals = {
            "szamla": f"SZ-{when.year}/{self.rng.randint(1, 9999):05d}",
            "kelt": format_date(when - timedelta(days=self.rng.randint(20, 90)), dstyle),
            "osszeg": format_amount(self.rng.randint(5, 900) * 1000),
            "hatarido": format_date(due, dstyle).rstrip("."),
            "bank": ids.generate_hu_account(self.rng, 24),
            "ugyszam": ref,
            "dok": self.choice(DOCS_MISSING),
        }
        y = m - 30
        c.text((m, y), sender, bold, s + 6)
        y += s + 16
        c.text((m, y), sender_addr, reg, s - 4, fill=(70, 70, 70))
        y += s
        sender_phone = self.phone()
        c.text((m, y), f"Tel.: {sender_phone}   E-mail: info@{strip_accents(sender.split()[0].lower())}.hu", reg, s - 4, fill=(70, 70, 70))
        y += s + 10
        c.hline(m, W, y, width=2, fill=(120, 120, 160))
        y += 50
        c.text((m, y), recipient, bold, s)
        y += s + 8
        for part in recipient_addr.split(", "):
            c.text((m, y), part, reg, s)
            y += s + 6
        y += 40
        date_line = f"{self.fake.city()}, {format_date(when, dstyle)}"
        c.text((W, y), date_line, reg, s, anchor="ra")
        y += s + 30
        c.text((m, y), f"Tárgy: {sc['targy']}", bold, s)
        y += s + 10
        c.text((m, y), f"Ügyszám: {ref}", reg, s)
        y += s + 40
        c.text((m, y), self.choice([f"Tisztelt {recipient}!", "Tisztelt Ügyfelünk!", "Tisztelt Címzett!"]), reg, s)
        y += s + 26
        for para in sc["body"]:
            for ln in c.wrap(para.format(**vals), reg, s, W - m):
                c.text((m, y), ln, reg, s)
                y += s + 10
            y += 14
        y += 30
        c.text((m, y), self.choice(["Üdvözlettel:", "Tisztelettel:"]), reg, s)
        y += s + 70
        signer = self.person()
        c.text((m + 40, y), signer, bold, s)
        y += s + 6
        c.text((m + 40, y), self.choice(["ügyvezető", "ügyintéző", "pénzügyi vezető", "irodavezető"]), reg, s - 2)
        fields = {
            "felado": sender, "felado_cim": sender_addr, "cimzett": recipient, "cimzett_cim": recipient_addr,
            "datum": format_date(when, dstyle), "targy": sc["targy"], "ugyszam": ref,
            "hatarido": vals["hatarido"], "teendo": sc["teendo"].format(**vals), "alairo": signer,
        }
        return Rendered(c.img, "level", fields, c.lines, [])

    def gen_oneletrajz(self, corrupt: bool = False) -> Rendered:
        c = Canvas(A4, self.rng)
        reg, bold = self.fonts(serif_ok=self.rng.random() < 0.3)
        s = self.rng.randint(21, 25)
        name = self.person()
        email = self.email(name)
        phone = self.phone()
        addr = self.address()
        birth = self.random_date(date(1975, 1, 1), date(2004, 12, 31))
        dstyle = self.choice([0, 1, 3])
        y0 = self.rng.randint(80, 110)
        sidebar = self.rng.random() < 0.45
        accent = self.choice([(30, 60, 120), (20, 20, 20), (120, 40, 40), (30, 100, 80)])
        schools = []
        uni, majors = self.choice(SCHOOLS)
        start = birth.year + 18 + self.rng.randint(0, 2)
        schools.append({"intezmeny": uni, "szak": self.choice(majors), "idoszak": f"{start} – {start + self.rng.randint(3, 5)}"})
        schools.append({"intezmeny": self.choice(HIGH_SCHOOLS), "szak": self.choice(["érettségi", "informatika tagozat", "általános tantervű osztály"]),
                        "idoszak": f"{start - 4} – {start}"})
        jobs = []
        jy = start + 4
        for _ in range(self.rng.randint(1, 3)):
            end = jy + self.rng.randint(1, 4)
            jobs.append({"ceg": self.company(), "pozicio": self.choice(POSITIONS),
                         "idoszak": f"{jy} – {end if end < 2026 else 'jelenleg'}"})
            jy = end
        jobs.reverse()
        langs = self.rng.sample(LANGUAGES, self.rng.randint(1, 2))
        skills = self.rng.sample(SKILLS, self.rng.randint(3, 6))
        contact = [("E-mail", email), ("Telefon", phone), ("Lakcím", addr), ("Születési dátum", format_date(birth, dstyle))]
        if sidebar:
            sw = 400
            c.draw.rectangle((0, 0, sw, A4[1]), fill=(235, 238, 243))
            x, y = 40, y0
            c.text((x, y), "ELÉRHETŐSÉG", bold, s, fill=accent)
            y += s + 20
            for k, v in contact:
                c.text((x, y), k, bold, s - 4)
                y += s
                for part in c.wrap(v, reg, s - 4, sw - 60):
                    c.text((x, y), part, reg, s - 4)
                    y += s
                y += 10
            y += 30
            c.text((x, y), "NYELVTUDÁS", bold, s, fill=accent)
            y += s + 16
            for l in langs:
                c.text((x, y), l, reg, s - 3)
                y += s + 4
            y += 30
            c.text((x, y), "KÉSZSÉGEK", bold, s, fill=accent)
            y += s + 16
            for sk in skills:
                c.text((x, y), sk, reg, s - 3)
                y += s + 4
            x, y, W = sw + 50, y0, A4[0] - 70
            c.text((x, y), name, bold, s + 22, fill=accent)
            y += s + 50
            c.text((x, y), self.choice(["Önéletrajz", "Szakmai önéletrajz", "Curriculum Vitae"]), reg, s + 2, fill=(90, 90, 90))
            y += s + 40
        else:
            x, W = 90, A4[0] - 90
            y = y0
            c.text((A4[0] // 2, y), name, bold, s + 24, anchor="ma", fill=accent)
            y += s + 50
            c.text((A4[0] // 2, y), self.choice(["Önéletrajz", "Szakmai önéletrajz", "Curriculum Vitae"]), reg, s + 2, anchor="ma", fill=(90, 90, 90))
            y += s + 30
            for k, v in contact:
                c.text((x, y), f"{k}:", bold, s - 2)
                c.text((x + 240, y), v, reg, s - 2)
                y += s + 8
            y += 20
        def section(title, y):
            c.text((x, y), title, bold, s + 2, fill=accent)
            c.hline(x, W, y + s + 12, width=2, fill=accent)
            return y + s + 28
        y = section("Szakmai tapasztalat", y)
        for j in jobs:
            c.text((x, y), j["idoszak"], reg, s - 2, fill=(90, 90, 90))
            y += s + 2
            c.text((x, y), f"{j['pozicio']} – {j['ceg']}", bold, s - 1)
            y += s + 18
        y += 10
        y = section("Tanulmányok", y)
        for sc in schools:
            c.text((x, y), sc["idoszak"], reg, s - 2, fill=(90, 90, 90))
            y += s + 2
            for part in c.wrap(sc["intezmeny"], bold, s - 1, W - x):
                c.text((x, y), part, bold, s - 1)
                y += s + 2
            c.text((x, y), sc["szak"], reg, s - 2)
            y += s + 18
        if not sidebar:
            y += 10
            y = section("Nyelvtudás", y)
            c.text((x, y), ", ".join(langs), reg, s - 2)
            y += s + 24
            y = section("Készségek", y)
            for part in c.wrap(", ".join(skills), reg, s - 2, W - x):
                c.text((x, y), part, reg, s - 2)
                y += s + 6
        fields = {
            "nev": name, "email": email, "telefon": phone, "lakcim": addr,
            "szuletesi_datum": format_date(birth, dstyle), "tanulmanyok": schools, "munkahelyek": jobs,
            "nyelvek": langs, "keszsegek": skills,
        }
        return Rendered(c.img, "oneletrajz", fields, c.lines, [])

    def gen_urlap(self, corrupt: bool = False) -> Rendered:
        c = Canvas(A4, self.rng)
        reg, bold = self.fonts()
        hand = self.choice(HAND) if self.rng.random() < 0.5 else reg
        hand_fill = self.choice([(20, 30, 120), (20, 20, 20), (30, 30, 90)])
        s = self.rng.randint(21, 24)
        m = self.rng.randint(80, 110)
        W = A4[0] - m
        title = self.choice(FORM_TITLES)
        name = self.person()
        mother = self.person()
        birth = self.random_date(date(1950, 1, 1), date(2006, 12, 31))
        dstyle = self.choice([0, 1])
        tax_id = ids.generate_tax_id(self.rng, birth)
        taj = ids.generate_taj(self.rng)
        errors = []
        if corrupt:
            kind = self.choice(["taj", "adoazonosito"])
            if kind == "taj":
                d = ids.digits(taj)
                taj = f"{d[:3]} {d[3:6]} {d[6:8]}{(int(d[8]) + self.rng.randint(1, 9)) % 10}"
                errors.append("taj_szam")
            else:
                tax_id = tax_id[:9] + str((int(tax_id[9]) + self.rng.randint(1, 9)) % 10)
                errors.append("adoazonosito_jel")
        signed = self.random_date()
        values = [
            ("Név", name), ("Anyja neve", mother), ("Születési hely", self.fake.city()),
            ("Születési idő", format_date(birth, dstyle)), ("Lakcím", self.address()), ("Telefonszám", self.phone()),
            ("E-mail cím", self.email(name)), ("Adóazonosító jel", tax_id), ("TAJ szám", taj),
        ]
        y = m
        c.text((A4[0] // 2, y), title, bold, s + 14, anchor="ma")
        y += s + 30
        c.text((A4[0] // 2, y), self.choice(["Kérjük, nyomtatott betűkkel töltse ki!", "A *-gal jelölt mezők kitöltése kötelező.", ""]),
               reg, s - 5, anchor="ma", fill=(90, 90, 90))
        y += s + 30
        c.text((m, y), "1. Személyes adatok", bold, s)
        y += s + 18
        label_w = self.rng.randint(280, 330)
        box_h = s + 30
        for label, value in values:
            c.rect((m, y, W, y + box_h), width=1, outline=(90, 90, 90))
            c.text((m + 12, y + box_h // 2), f"{label}:", reg, s - 3, anchor="lm")
            c.draw.line([(m + label_w, y), (m + label_w, y + box_h)], fill=(140, 140, 140), width=1)
            c.text((m + label_w + 16, y + box_h // 2 + self.rng.randint(-3, 3)), value, hand, s + (2 if hand != reg else 0), fill=hand_fill, anchor="lm")
            y += box_h
        y += 40
        c.text((m, y), "2. Nyilatkozatok", bold, s)
        y += s + 20
        for stmt in ["Hozzájárulok adataim kezeléséhez.", "Hírlevelet kérek.", "Az adatok a valóságnak megfelelnek."]:
            c.rect((m, y, m + 28, y + 28), width=2)
            if self.rng.random() < 0.7:
                c.draw.line([(m + 5, y + 14), (m + 12, y + 23), (m + 25, y + 4)], fill=hand_fill, width=3)
            c.text((m + 45, y + 14), stmt, reg, s - 3, anchor="lm")
            y += 45
        y += 60
        kelt = f"{self.fake.city()}, {format_date(signed, dstyle)}"
        c.text((m, y), "Kelt:", reg, s - 2)
        c.text((m + 90, y - 4), kelt, hand, s, fill=hand_fill)
        c.hline(W - 380, W, y + 40, width=1)
        c.text((W - 190, y + 50), "aláírás", reg, s - 6, anchor="ma", fill=(90, 90, 90))
        fields = {
            "urlap_cime": title, "nev": values[0][1], "anyja_neve": values[1][1], "szuletesi_hely": values[2][1],
            "szuletesi_datum": values[3][1], "lakcim": values[4][1], "telefon": values[5][1], "email": values[6][1],
            "adoazonosito_jel": values[7][1], "taj_szam": values[8][1], "kelt": kelt,
        }
        return Rendered(c.img, "urlap", fields, c.lines, errors)


def scan_effect(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-1.5, 1.5)
    out = img.convert("L").rotate(angle, expand=True, fillcolor=rng.randint(235, 255), resample=Image.BICUBIC)
    arr = np.asarray(out).astype(np.float32)
    arr += np.random.default_rng(rng.randint(0, 10**9)).normal(0, rng.uniform(2, 8), arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    return _jpeg(out.convert("RGB"), rng.randint(60, 92))


def photo_effect(img: Image.Image, rng: random.Random) -> Image.Image:
    import cv2
    w, h = img.size
    pad = int(max(w, h) * rng.uniform(0.08, 0.2))
    bg_color = np.array(rng.choice([(110, 85, 60), (70, 70, 75), (150, 140, 120), (40, 45, 55), (180, 175, 165)]), np.float32)
    nrng = np.random.default_rng(rng.randint(0, 10**9))
    H, Wd = h + 2 * pad, w + 2 * pad
    bg = np.ones((H, Wd, 3), np.float32) * bg_color + nrng.normal(0, 12, (H, Wd, 3))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = lambda: rng.uniform(-0.06, 0.06) * max(w, h)
    dst = np.float32([[pad + j(), pad + j()], [pad + w + j(), pad + j()], [pad + w + j(), pad + h + j()], [pad + j(), pad + h + j()]])
    M = cv2.getPerspectiveTransform(src, dst)
    doc = np.asarray(img.convert("RGB")).astype(np.float32)
    warped = cv2.warpPerspective(doc, M, (Wd, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (Wd, H))
    out = bg * (1 - mask[..., None]) + warped * mask[..., None]
    gx = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.85, 1.1), Wd)[None, :]
    gy = np.linspace(rng.uniform(0.7, 1.05), rng.uniform(0.8, 1.1), H)[:, None]
    out *= (gx * gy)[..., None]
    out *= np.array([rng.uniform(0.92, 1.05), rng.uniform(0.95, 1.02), rng.uniform(0.85, 1.0)], np.float32)
    out += nrng.normal(0, rng.uniform(2, 7), out.shape)
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if rng.random() < 0.7:
        res = res.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
    scale = rng.uniform(0.75, 1.0)
    res = res.resize((int(res.width * scale), int(res.height * scale)), Image.BILINEAR)
    return _jpeg(res, rng.randint(55, 90))


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_rendered(r: Rendered, out_dir: Path, stem: str, variant: str, rng: random.Random) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if variant == "photo":
        img = photo_effect(r.image, rng)
    elif variant == "scan":
        img = scan_effect(r.image, rng)
    else:
        img = r.image
    path = out_dir / f"{stem}.jpg"
    img.save(path, "JPEG", quality=92)
    meta = {"doc_type": r.doc_type, "variant": variant, "fields": r.fields, "lines": r.lines, "errors": r.errors,
            "size": list(r.image.size)}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def ground_truth_text(meta: dict) -> str:
    lines = sorted(meta["lines"], key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return "\n".join(l["text"] for l in lines)


def ground_truth_lines(meta: dict):
    from .ocr import Segment, group_lines
    return group_lines([Segment(l["text"], *l["bbox"], 1.0) for l in meta["lines"]])


def ground_truth_fields(meta: dict) -> dict:
    from .schemas import scalar_fields
    scalars = set(scalar_fields(meta["doc_type"]))
    return {k: ({"ertek": v, "sor": None} if k in scalars else v) for k, v in meta["fields"].items()}
