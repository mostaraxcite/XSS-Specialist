"""Deterministic structural taxonomy for XSS semantic fields.

This module classifies only explicit source/sink/defense field expressions.
It does not decide whether a vulnerability exists and never confirms XSS.
"""
from __future__ import annotations
import re


def _s(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def classify_source(expression: str) -> dict:
    low=_s(expression)
    if not low:
        return {"label":"NONE","matched":"empty"}
    browser=(
        "location.hash","location.search","window.location.hash","window.location.search",
        "document.referrer","document.url","document.documenturi","document.cookie",
        "window.name","localstorage","sessionstorage","event.data","evt.data","message.data",
    )
    server=("req.query","req.body","req.params","request.query","request.body","request.params","$_get","$_post")
    framework=("props.","route.params","router.query","parammap","ctx.params","pageprops","params.","_ctx.")
    if any(x in low for x in browser):
        return {"label":"BROWSER","matched":"known_browser_source"}
    if any(x in low for x in server):
        return {"label":"SERVER","matched":"known_server_source"}
    if any(x in low for x in framework):
        return {"label":"FRAMEWORK","matched":"known_framework_source"}
    if (
        re.fullmatch(r"(?:['\"].*['\"]|\d+(?:\.\d+)?|true|false|null|undefined)",low)
        or low.startswith(("static_","default_","config.","settings."))
    ):
        return {"label":"NONE","matched":"literal_or_static"}
    return {"label":"OTHER","matched":"unrecognized_source"}


def classify_sink(expression: str) -> dict:
    low=_s(expression)
    html=("innerhtml","outerhtml","document.write","document.writeln","insertadjacenthtml",".html(","dangerouslysetinnerhtml","res.send(","v-html=","x-html","unsafehtml(")
    js=("eval(","function(","new function(","settimeout(","setinterval(")
    url=(".href =", ".src =", ".action =", ".formaction =", "object.data =","res.redirect(")
    safe=("textcontent","innertext","insertadjacenttext",".text(","createtextnode","nodevalue","v-text=","x-text")
    if any(x in low for x in html):
        return {"label":"DANGEROUS_HTML","matched":"known_html_sink"}
    if any(x in low for x in js):
        return {"label":"DANGEROUS_JS","matched":"known_js_sink"}
    if any(x in low for x in url):
        return {"label":"DANGEROUS_URL","matched":"known_url_sink"}
    if any(x in low for x in safe):
        return {"label":"SAFE_OUTPUT","matched":"known_safe_output"}
    if low.startswith(("return ","json.parse(","json.stringify(","const ","let ","items.push(","store.push(","cache.set(")):
        return {"label":"NONE","matched":"non_sink_operation"}
    return {"label":"OTHER","matched":"unrecognized_sink"}


def classify_defense(expression: str) -> dict:
    low=_s(expression)
    if low in {"[no_defense]","[no_protection]","direct_to_sink","direct_use","unsanitized","no_transform","raw_value"}:
        return {"label":"NONE","matched":"explicit_absence"}
    if any(x in low for x in ("dompurify.sanitize","sanitizehtml(","purifier.sanitize","htmlsanitizer.clean","xssfilter.process","sanitize(")):
        return {"label":"SANITIZATION","matched":"known_sanitizer"}
    if any(x in low for x in ("escapehtml","htmlspecialchars","encodeforhtml","escapeattribute","htmlencode","encodeforattribute","encodeuri(","encodeuricomponent(","escapeexpression(")):
        return {"label":"CONTEXTUAL_ENCODING","matched":"known_encoder"}
    if any(x in low for x in ("textcontent","innertext","insertadjacenttext",".text(","createtextnode")):
        return {"label":"SAFE_DOM_API","matched":"safe_dom_api"}
    if any(x in low for x in ("allowedprotocol","valid_schemes","validator.isurl","startswith('https://')","protocol ===","includes(new url","match(/^https")):
        return {"label":"INPUT_CONSTRAINT","matched":"input_constraint"}
    if (
        ("{" in expression and "}" in expression and "<" in expression and ">" in expression)
        or "{{" in expression
        or "v-text=" in low
        or "x-text" in low
        or "safeinterpolation(" in low
        or "rendertext(" in low
    ):
        return {"label":"FRAMEWORK_ESCAPING","matched":"framework_text_escaping"}
    return {"label":"OTHER","matched":"unrecognized_defense"}
