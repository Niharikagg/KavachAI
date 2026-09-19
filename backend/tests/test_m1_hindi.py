import pytest

from backend.modules.nlp.detector import detect_entities
from backend.modules.nlp.extractor import _find_age

HI = {"observed_scripts": ["hi"], "is_code_mixed": False}

PARAGRAPH = (
    "मेरे पिता का नाम रमेश शर्मा है। वे 52 वर्ष के हैं और बेंगलुरु, कर्नाटक में रहते हैं। "
    "वे एक सरकारी विद्यालय में वरिष्ठ गणित शिक्षक के रूप में काम करते हैं और पिछले 25 वर्षों से "
    "अध्यापन के क्षेत्र में हैं। उन्होंने गणित में स्नातकोत्तर की पढ़ाई पूरी की है और शिक्षा के "
    "क्षेत्र में कई प्रशिक्षण कार्यक्रमों में भाग लिया है। उन्हें उच्च रक्तचाप की समस्या है और वे "
    "नियमित रूप से अपने डॉक्टर से परामर्श लेते हैं। वे अपने परिवार के साथ रहते हैं और अपने बच्चों "
    "की शिक्षा को बहुत महत्व देते हैं। अच्छी शिक्षा और अनुशासन व्यक्ति के जीवन में बहुत महत्वपूर्ण "
    "भूमिका निभाते हैं।"
)

NO_PERSON = [
    "वे एक सरकारी विद्यालय में वरिष्ठ गणित शिक्षक के रूप में काम करते हैं।",
    "उन्होंने गणित में स्नातकोत्तर की पढ़ाई पूरी की है।",
    "शिक्षा के क्षेत्र में कई प्रशिक्षण कार्यक्रमों में भाग लिया है।",
    "उन्हें उच्च रक्तचाप की समस्या है।",
    "वे अपने परिवार के साथ रहते हैं और अपने बच्चों की शिक्षा को महत्व देते हैं।",
    "अच्छी शिक्षा और अनुशासन व्यक्ति के जीवन में महत्वपूर्ण हैं।",
    "वे अपने डॉक्टर से परामर्श लेते हैं।",
]


@pytest.mark.parametrize("t", NO_PERSON)
def test_no_person_false_positive(t):
    assert not [e for e in detect_entities(t, HI) if e["type"] == "PERSON"]


@pytest.mark.parametrize("t,name", [
    ("मेरे पिता का नाम रमेश शर्मा है।", "रमेश शर्मा"),
    ("मेरा नाम राहुल शर्मा है और मेरा फोन नंबर 9876543210 है।", "राहुल शर्मा"),
    ("उनका नाम अमित कुमार है।", "अमित कुमार"),
])
def test_person_true_positive(t, name):
    assert name in [e["value"] for e in detect_entities(t, HI) if e["type"] == "PERSON"]


def test_full_paragraph_only_one_person():
    persons = [e["value"] for e in detect_entities(PARAGRAPH, HI) if e["type"] == "PERSON"]
    assert persons == ["रमेश शर्मा"]


@pytest.mark.parametrize("t,expected", [
    ("वे 52 वर्ष के हैं", ["52"]),
    ("मेरी उम्र 30 है", ["30"]),
    ("वह 40 वर्षीय शिक्षक है", ["40"]),
    ("पिछले 25 वर्षों से अध्यापन के क्षेत्र में हैं", []),
    ("उन्हें 25 वर्ष का अनुभव है", []),
])
def test_age(t, expected):
    assert [a["value"] for a in _find_age(t)] == expected