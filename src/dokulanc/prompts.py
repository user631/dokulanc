SYSTEM = (
    "Magyar nyelvű dokumentumok adatkinyerő rendszere vagy. A bemenet egy OCR-rel felismert dokumentum "
    "számozott sorai; a sorokban a ' | ' jel egymástól távol eső szövegrészeket (oszlopokat) választ el. "
    "Az OCR hibázhat, ezért a sorok sorrendje és a karakterek nem mindig tökéletesek. "
    "Szabályok: 1) Csak a szövegben ténylegesen szereplő adatot adj vissza, ne találj ki semmit. "
    "2) Ha egy adat nem szerepel, az értéke legyen null. "
    "3) Az értékeket pontosan úgy add vissza, ahogy a szövegben állnak (ne alakítsd át a dátumot, összeget, számot). "
    "4) Minden mezőnél add meg annak a sornak a számát, ahonnan az értéket vetted. "
    "5) A címkéket (pl. 'Adószám:') ne tedd bele az értékbe. "
    "Kizárólag a megadott JSON-sémának megfelelő választ adj."
)

TYPE_HINTS = {
    "szamla": (
        "A dokumentum egy SZÁMLA. Gyűjtsd ki a számla sorszámát, a kiállítás, teljesítés dátumát és a fizetési "
        "határidőt, az eladó és a vevő nevét, címét, adószámát, a bankszámlaszámot, a fizetési módot, a nettó, ÁFA és "
        "bruttó végösszeget, valamint a tételeket (megnevezés, mennyiség egységgel, egységár, nettó, ÁFA-kulcs, "
        "ÁFA-összeg, bruttó). Az eladó és a vevő adatait ne keverd össze: az 'Eladó'/'Szállító' címke alatti vagy melletti "
        "blokk az eladóé, a 'Vevő' címke alatti a vevőé."
    ),
    "nyugta": (
        "A dokumentum egy NYUGTA (pénztárgépes blokk). Gyűjtsd ki az üzlet nevét, címét, adószámát, a vásárlás "
        "dátumát és időpontját, a nyugtaszámot, a fizetési módot, a végösszeget és a tételeket (megnevezés, összeg). "
        "A tétel megnevezése a sor eleji szöveg, az összege a sor végi szám az ÁFA-kód (pl. C00) nélkül. "
        "Az 'N DB x ár' sorok nem külön tételek, ezeket hagyd figyelmen kívül. "
        "Példa: a '3 DB x 250' és 'ZSEMLE | 750 C00' sorokból egyetlen tétel lesz: megnevezes='ZSEMLE', osszeg='750'."
    ),
    "level": (
        "A dokumentum egy hivatalos LEVÉL. Gyűjtsd ki a feladót (cég vagy személy) és címét, a címzettet és címét, "
        "a levél keltezésének dátumát (helységnév nélkül), a tárgyat, az ügyszámot vagy hivatkozási számot, a levélben "
        "szereplő határidőt, az aláíró nevét, majd saját szavaiddal írd le, mit kell a címzettnek tennie (teendo), "
        "és foglald össze a levelet 1-2 mondatban (osszefoglalo)."
    ),
    "oneletrajz": (
        "A dokumentum egy ÖNÉLETRAJZ. Gyűjtsd ki a nevet, e-mail-címet, telefonszámot, lakcímet, születési dátumot, "
        "a tanulmányokat (intézmény, szak vagy végzettség, időszak), a munkahelyeket (cég, pozíció, időszak), "
        "a nyelvtudást és a készségeket. A tanulmányok listájába minden felsorolt iskola kerüljön bele, a középiskola "
        "(gimnázium, technikum) is; minden bejegyzésnél az a hozzá tartozó időszak legyen, amely közvetlenül fölötte "
        "vagy mellette áll. A munkahelyeknél a 'pozíció – cég' sorból a pozíciót és a céget külön mezőbe tedd."
    ),
    "urlap": (
        "A dokumentum egy kitöltött ŰRLAP. Az űrlap címe a lap tetején álló cím. A mezők címkéje és a kitöltött érték "
        "általában egy sorban van ' | ' jellel elválasztva. Gyűjtsd ki a kitöltő nevét, anyja nevét, születési helyét és "
        "idejét, lakcímét, telefonszámát, e-mail-címét, adóazonosító jelét, TAJ-számát és a keltezést (helységgel együtt)."
    ),
}

CLASSIFY = (
    "Döntsd el a dokumentum szövege alapján, hogy milyen típusú: szamla (számla), nyugta (pénztárgépes blokk), "
    "level (hivatalos levél), oneletrajz (önéletrajz) vagy urlap (kitöltött űrlap/adatlap). "
    "Röviden indokold a döntést."
)


def format_lines(lines) -> str:
    return "\n".join(f"[{line.id}] {line.text}" for line in lines)


def extraction_messages(doc_type: str, lines) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{TYPE_HINTS[doc_type]}\n\nA dokumentum sorai:\n{format_lines(lines)}"},
    ]


def classify_messages(lines) -> list[dict]:
    return [
        {"role": "system", "content": "Dokumentumtípus-felismerő rendszer vagy. Kizárólag JSON-ban válaszolj."},
        {"role": "user", "content": f"{CLASSIFY}\n\nA dokumentum sorai:\n{format_lines(lines)}"},
    ]
