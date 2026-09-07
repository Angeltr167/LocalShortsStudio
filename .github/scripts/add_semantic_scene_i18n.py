import json
from pathlib import Path


TRANSLATIONS = {
    "en": (
        "Semantic Scene Ranking",
        "Use the local OpenCLIP ranker to score stock-video preview images against each scene query before downloading the clip. Requires Strict Scene Matching. If the local ranker is unavailable, provider order is used as a safe fallback.",
    ),
    "de": (
        "Semantische Szenenbewertung",
        "Bewertet Stockvideo-Vorschaubilder mit dem lokalen OpenCLIP-Ranker gegen jede Szenenanfrage, bevor der Clip heruntergeladen wird. Erfordert die strikte Szenenzuordnung. Ist der lokale Ranker nicht verfügbar, bleibt die Anbieterreihenfolge als sichere Ausweichlösung erhalten.",
    ),
    "es": (
        "Ranking semántico de escenas",
        "Usa el ranker local OpenCLIP para puntuar las imágenes de vista previa de los videos de stock frente a cada búsqueda de escena antes de descargar el clip. Requiere Coincidencia estricta por escenas. Si el ranker local no está disponible, se conserva el orden del proveedor como respaldo seguro.",
    ),
    "fr": (
        "Classement sémantique des scènes",
        "Utilise le ranker OpenCLIP local pour comparer les aperçus des vidéos stock à chaque requête de scène avant de télécharger le clip. Nécessite la correspondance stricte des scènes. Si le ranker local est indisponible, l’ordre du fournisseur est conservé comme solution de repli sûre.",
    ),
    "id": (
        "Peringkat Semantik Adegan",
        "Menggunakan ranker OpenCLIP lokal untuk menilai gambar pratinjau video stok terhadap setiap kueri adegan sebelum klip diunduh. Memerlukan Pencocokan Adegan Ketat. Jika ranker lokal tidak tersedia, urutan penyedia tetap digunakan sebagai fallback aman.",
    ),
    "it": (
        "Classifica semantica delle scene",
        "Usa il ranker OpenCLIP locale per valutare le anteprime dei video stock rispetto a ogni query di scena prima di scaricare il clip. Richiede la corrispondenza rigorosa delle scene. Se il ranker locale non è disponibile, viene mantenuto l’ordine del provider come fallback sicuro.",
    ),
    "ko": (
        "장면 의미 순위 지정",
        "클립을 다운로드하기 전에 로컬 OpenCLIP 랭커로 각 장면 쿼리와 스톡 비디오 미리보기 이미지를 비교해 점수를 매깁니다. 엄격한 장면 매칭이 필요합니다. 로컬 랭커를 사용할 수 없으면 공급자 순서를 안전한 대체 경로로 유지합니다.",
    ),
    "pt": (
        "Ranking semântico de cenas",
        "Usa o ranker OpenCLIP local para pontuar as prévias dos vídeos de stock em relação a cada consulta de cena antes de baixar o clipe. Requer Correspondência estrita de cenas. Se o ranker local não estiver disponível, a ordem do provedor é mantida como fallback seguro.",
    ),
    "ru": (
        "Семантическое ранжирование сцен",
        "Использует локальный ранкер OpenCLIP для оценки превью стоковых видео относительно запроса каждой сцены до загрузки клипа. Требует строгого сопоставления сцен. Если локальный ранкер недоступен, порядок провайдера сохраняется как безопасный резервный вариант.",
    ),
    "tr": (
        "Semantik Sahne Sıralaması",
        "Klibi indirmeden önce her sahne sorgusuna göre stok video önizlemelerini yerel OpenCLIP sıralayıcısıyla puanlar. Sıkı Sahne Eşleştirme gerektirir. Yerel sıralayıcı kullanılamazsa güvenli geri dönüş olarak sağlayıcı sırası korunur.",
    ),
    "vi": (
        "Xếp hạng ngữ nghĩa cảnh",
        "Dùng bộ xếp hạng OpenCLIP cục bộ để chấm điểm ảnh xem trước video stock so với từng truy vấn cảnh trước khi tải clip. Yêu cầu Khớp cảnh nghiêm ngặt. Nếu bộ xếp hạng cục bộ không khả dụng, thứ tự của nhà cung cấp được giữ làm phương án dự phòng an toàn.",
    ),
    "zh": (
        "场景语义排序",
        "在下载片段前，使用本地 OpenCLIP 排序服务将库存视频预览图与每个场景查询进行语义评分。需要启用严格场景匹配。如果本地排序服务不可用，将安全回退到素材供应商原始顺序。",
    ),
}


def append_keys(locale: str, label: str, help_text: str) -> None:
    path = Path("webui/i18n") / f"{locale}.json"
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    translation_map = data.get("Translation")
    if not isinstance(translation_map, dict):
        raise RuntimeError(f"missing Translation map in {locale}")

    expected = {
        "Semantic Scene Ranking": label,
        "Semantic Scene Ranking Help": help_text,
    }
    if all(translation_map.get(key) == value for key, value in expected.items()):
        return
    if any(key in translation_map for key in expected):
        raise RuntimeError(f"partial or conflicting semantic-scene translation in {locale}")

    stripped = text.rstrip()
    ending = "\n  }\n}"
    if not stripped.endswith(ending):
        raise RuntimeError(f"unexpected JSON ending in {path}")
    body = stripped[: -len(ending)].rstrip()
    if body and not body.endswith(","):
        body += ","
    additions = (
        "\n"
        f"    {json.dumps('Semantic Scene Ranking')}: {json.dumps(label, ensure_ascii=False)},\n"
        f"    {json.dumps('Semantic Scene Ranking Help')}: {json.dumps(help_text, ensure_ascii=False)}"
        "\n  }\n}\n"
    )
    updated = body + additions
    parsed = json.loads(updated)
    parsed_map = parsed.get("Translation") or {}
    for key, value in expected.items():
        if parsed_map.get(key) != value:
            raise RuntimeError(f"failed to add {key!r} to {locale}")
    path.write_text(updated, encoding="utf-8")


for locale, (label, help_text) in TRANSLATIONS.items():
    append_keys(locale, label, help_text)

print("semantic scene translations added")
