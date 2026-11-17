from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

Szoveg = Annotated[str, Field(max_length=200)]
Nev = Annotated[str, Field(max_length=120)]
Szam = Annotated[str, Field(max_length=30)]
Hosszu = Annotated[str, Field(max_length=600)]
MAX_ITEMS = 40


class Mezo(BaseModel):
    ertek: Optional[Szoveg] = Field(None, description="Az érték pontosan úgy, ahogy a szövegben szerepel; null, ha nincs")
    sor: Optional[int] = Field(None, description="Annak a sornak a száma, ahonnan az érték származik")


class SzamlaTetel(BaseModel):
    megnevezes: Optional[Nev] = None
    mennyiseg: Optional[Szam] = None
    egysegar: Optional[Szam] = None
    netto: Optional[Szam] = None
    afa_kulcs: Optional[Szam] = None
    afa: Optional[Szam] = None
    brutto: Optional[Szam] = None
    sor: Optional[int] = None


class Szamla(BaseModel):
    szamlaszam: Mezo
    kiallitas_datuma: Mezo
    teljesites_datuma: Mezo
    fizetesi_hatarido: Mezo
    elado_nev: Mezo
    elado_cim: Mezo
    elado_adoszam: Mezo
    vevo_nev: Mezo
    vevo_cim: Mezo
    vevo_adoszam: Mezo
    bankszamla: Mezo
    fizetesi_mod: Mezo
    netto_osszesen: Mezo
    afa_osszesen: Mezo
    brutto_osszesen: Mezo
    tetelek: list[SzamlaTetel] = Field(max_length=MAX_ITEMS)


class NyugtaTetel(BaseModel):
    megnevezes: Optional[Nev] = None
    osszeg: Optional[Szam] = None
    sor: Optional[int] = None


class Nyugta(BaseModel):
    uzlet_nev: Mezo
    uzlet_cim: Mezo
    adoszam: Mezo
    datum: Mezo
    ido: Mezo
    nyugtaszam: Mezo
    fizetesi_mod: Mezo
    vegosszeg: Mezo
    tetelek: list[NyugtaTetel] = Field(max_length=MAX_ITEMS)


class Level(BaseModel):
    felado: Mezo
    felado_cim: Mezo
    cimzett: Mezo
    cimzett_cim: Mezo
    datum: Mezo
    targy: Mezo
    ugyszam: Mezo
    hatarido: Mezo
    alairo: Mezo
    teendo: Optional[Hosszu] = Field(None, description="Mit kell a címzettnek tennie, saját szavakkal, egy mondatban")
    osszefoglalo: Optional[Hosszu] = Field(None, description="A levél lényege 1-2 mondatban")


class Tanulmany(BaseModel):
    intezmeny: Optional[Nev] = None
    szak: Optional[Nev] = None
    idoszak: Optional[Nev] = None
    sor: Optional[int] = None


class Munkahely(BaseModel):
    ceg: Optional[Nev] = None
    pozicio: Optional[Nev] = None
    idoszak: Optional[Nev] = None
    sor: Optional[int] = None


class Oneletrajz(BaseModel):
    nev: Mezo
    email: Mezo
    telefon: Mezo
    lakcim: Mezo
    szuletesi_datum: Mezo
    tanulmanyok: list[Tanulmany] = Field(max_length=MAX_ITEMS)
    munkahelyek: list[Munkahely] = Field(max_length=MAX_ITEMS)
    nyelvek: list[Nev] = Field(max_length=MAX_ITEMS)
    keszsegek: list[Nev] = Field(max_length=MAX_ITEMS)


class Urlap(BaseModel):
    urlap_cime: Mezo
    nev: Mezo
    anyja_neve: Mezo
    szuletesi_hely: Mezo
    szuletesi_datum: Mezo
    lakcim: Mezo
    telefon: Mezo
    email: Mezo
    adoazonosito_jel: Mezo
    taj_szam: Mezo
    kelt: Mezo


class TipusValasz(BaseModel):
    tipus: Literal["szamla", "nyugta", "level", "oneletrajz", "urlap"]
    indoklas: Hosszu


SCHEMAS: dict[str, type[BaseModel]] = {
    "szamla": Szamla,
    "nyugta": Nyugta,
    "level": Level,
    "oneletrajz": Oneletrajz,
    "urlap": Urlap,
}

LIST_FIELDS = {"tetelek", "tanulmanyok", "munkahelyek", "nyelvek", "keszsegek"}
FREE_TEXT_FIELDS = {"teendo", "osszefoglalo"}


def scalar_fields(doc_type: str) -> list[str]:
    model = SCHEMAS[doc_type]
    return [name for name, f in model.model_fields.items() if f.annotation is Mezo]


def inline_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            out = {k: resolve(v) for k, v in node.items() if k not in ("title", "default")}
            if out.get("type") == "object" and "properties" in out:
                out["required"] = list(out["properties"])
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def empty_result(doc_type: str) -> dict:
    model = SCHEMAS[doc_type]
    out = {}
    for name, f in model.model_fields.items():
        if f.annotation is Mezo:
            out[name] = {"ertek": None, "sor": None}
        elif name in LIST_FIELDS:
            out[name] = []
        else:
            out[name] = None
    return out
