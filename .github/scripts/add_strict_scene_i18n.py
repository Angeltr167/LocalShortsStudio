import json
from pathlib import Path


TRANSLATIONS = {
    "de": (
        "Strikte Szenenzuordnung",
        "Behandelt geordnete Stockvideo-Schlüsselwörter als Szenensuchen, weist jeder Timeline-Szene einen Clip zu, vermeidet nach Möglichkeit doppelte Quellen und bewahrt die Erzählreihenfolge. Verfügbar für Pexels, Pixabay und Coverr.",
    ),
    "es": (
        "Coincidencia estricta por escenas",
        "Trata las palabras clave ordenadas como búsquedas de escena, asigna un clip por escena de la línea de tiempo, evita fuentes duplicadas cuando sea posible y conserva el orden narrativo. Disponible para Pexels, Pixabay y Coverr.",
    ),
    "fr": (
        "Correspondance stricte des scènes",
        "Traite les mots-clés de vidéos de stock ordonnés comme des recherches de scène, attribue un clip à chaque scène de la timeline, évite les sources en double si possible et préserve l’ordre narratif. Disponible pour Pexels, Pixabay et Coverr.",
    ),
    "id": (
        "Pencocokan Adegan Ketat",
        "Memperlakukan kata kunci video stok yang berurutan sebagai pencarian adegan, menetapkan satu klip per adegan timeline, menghindari sumber duplikat bila memungkinkan, dan mempertahankan urutan narasi. Tersedia untuk Pexels, Pixabay, dan Coverr.",
    ),
    "it": (
        "Corrispondenza rigorosa delle scene",
        "Tratta le parole chiave ordinate dei video stock come ricerche di scena, assegna un clip a ogni scena della timeline, evita fonti duplicate quando possibile e mantiene l’ordine narrativo. Disponibile per Pexels, Pixabay e Coverr.",
    ),
    "ko": (
        "엄격한 장면 매칭",
        "정렬된 스톡 비디오 키워드를 장면 검색어로 사용하고, 타임라인의 각 장면에 클립 하나를 할당하며, 가능한 경우 중복 소스를 피하고 서사 순서를 유지합니다. Pexels, Pixabay 및 Coverr에서 사용할 수 있습니다.",
    ),
    "pt": (
        "Correspondência estrita de cenas",
        "Trata palavras-chave ordenadas de vídeos de stock como buscas de cena, atribui um clipe a cada cena da linha do tempo, evita fontes duplicadas quando possível e preserva a ordem narrativa. Disponível para Pexels, Pixabay e Coverr.",
    ),
    "ru": (
        "Строгое сопоставление сцен",
        "Использует упорядоченные ключевые слова стокового видео как запросы для сцен, назначает по одному клипу каждой сцене временной шкалы, по возможности избегает повторов источников и сохраняет порядок повествования. Доступно для Pexels, Pixabay и Coverr.",
    ),
    "tr": (
        "Sıkı Sahne Eşleştirme",
        "Sıralı stok video anahtar kelimelerini sahne sorguları olarak kullanır, zaman çizelgesindeki her sahneye bir klip atar, mümkün olduğunda yinelenen kaynaklardan kaçınır ve anlatı sırasını korur. Pexels, Pixabay ve Coverr için kullanılabilir.",
    ),
    "vi": (
        "Khớp cảnh nghiêm ngặt",
        "Dùng các từ khóa video stock theo thứ tự làm truy vấn cảnh, gán một clip cho mỗi cảnh trên timeline, tránh nguồn trùng lặp khi có thể và giữ nguyên thứ tự nội dung. Hỗ trợ Pexels, Pixabay và Coverr.",
    ),
}


def append_keys(locale: str, label: str, help_text: str) -> None:
    path = Path("webui/i18n") / f"{locale}.json"
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    expected = {
        "Strict Scene Matching": label,
        "Strict Scene Matching Help": help_text,
    }
    if all(data.get(key) == value for key, value in expected.items()):
        return
    if any(key in data for key in expected):
        raise RuntimeError(f"partial or conflicting strict-scene translation in {locale}")

    stripped = text.rstrip()
    if not stripped.endswith("}"):
        raise RuntimeError(f"unexpected JSON ending in {path}")
    body = stripped[:-1].rstrip()
    if body and not body.endswith(","):
        body += ","
    additions = (
        "\n"
        f"    {json.dumps('Strict Scene Matching')}: "
        f"{json.dumps(label, ensure_ascii=False)},\n"
        f"    {json.dumps('Strict Scene Matching Help')}: "
        f"{json.dumps(help_text, ensure_ascii=False)}\n"
        "}\n"
    )
    updated = body + additions
    parsed = json.loads(updated)
    for key, value in expected.items():
        if parsed.get(key) != value:
            raise RuntimeError(f"failed to add {key!r} to {locale}")
    path.write_text(updated, encoding="utf-8")


for locale, (label, help_text) in TRANSLATIONS.items():
    append_keys(locale, label, help_text)

print("strict scene translations added to all required secondary locales")
