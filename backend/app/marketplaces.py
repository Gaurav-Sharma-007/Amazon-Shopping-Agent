from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AmazonMarketplace:
    code: str
    country: str
    domain: str
    region: str
    currency_code: str
    currency_symbol: str
    aliases: tuple[str, ...]


AMAZON_MARKETPLACES: tuple[AmazonMarketplace, ...] = (
    AmazonMarketplace("US", "United States", "https://www.amazon.com", "Americas", "USD", "$", ("united states", "usa", "us", "america", "amazon.com")),
    AmazonMarketplace("CA", "Canada", "https://www.amazon.ca", "Americas", "CAD", "C$", ("canada", "ca", "amazon.ca")),
    AmazonMarketplace("MX", "Mexico", "https://www.amazon.com.mx", "Americas", "MXN", "MX$", ("mexico", "mx", "amazon.com.mx")),
    AmazonMarketplace("BR", "Brazil", "https://www.amazon.com.br", "Americas", "BRL", "R$", ("brazil", "brasil", "br", "amazon.com.br")),
    AmazonMarketplace("UK", "United Kingdom", "https://www.amazon.co.uk", "Europe", "GBP", "£", ("united kingdom", "uk", "great britain", "britain", "england", "amazon.co.uk")),
    AmazonMarketplace("FR", "France", "https://www.amazon.fr", "Europe", "EUR", "€", ("france", "fr", "amazon.fr")),
    AmazonMarketplace("BE", "Belgium", "https://www.amazon.com.be", "Europe", "EUR", "€", ("belgium", "be", "amazon.com.be")),
    AmazonMarketplace("ES", "Spain", "https://www.amazon.es", "Europe", "EUR", "€", ("spain", "es", "amazon.es")),
    AmazonMarketplace("DE", "Germany", "https://www.amazon.de", "Europe", "EUR", "€", ("germany", "deutschland", "de", "amazon.de")),
    AmazonMarketplace("IE", "Ireland", "https://www.amazon.ie", "Europe", "EUR", "€", ("ireland", "ie", "amazon.ie")),
    AmazonMarketplace("IT", "Italy", "https://www.amazon.it", "Europe", "EUR", "€", ("italy", "it", "amazon.it")),
    AmazonMarketplace("NL", "Netherlands", "https://www.amazon.nl", "Europe", "EUR", "€", ("netherlands", "holland", "nl", "amazon.nl")),
    AmazonMarketplace("PL", "Poland", "https://www.amazon.pl", "Europe", "PLN", "zł", ("poland", "pl", "amazon.pl")),
    AmazonMarketplace("SE", "Sweden", "https://www.amazon.se", "Europe", "SEK", "kr", ("sweden", "se", "amazon.se")),
    AmazonMarketplace("TR", "Turkey", "https://www.amazon.com.tr", "Europe", "TRY", "₺", ("turkey", "türkiye", "turkiye", "tr", "amazon.com.tr")),
    AmazonMarketplace("AU", "Australia", "https://www.amazon.com.au", "Asia-Pacific", "AUD", "A$", ("australia", "au", "amazon.com.au")),
    AmazonMarketplace("IN", "India", "https://www.amazon.in", "Asia-Pacific", "INR", "₹", ("india", "in", "bharat", "amazon.in")),
    AmazonMarketplace("JP", "Japan", "https://www.amazon.co.jp", "Asia-Pacific", "JPY", "¥", ("japan", "jp", "amazon.co.jp")),
    AmazonMarketplace("SG", "Singapore", "https://www.amazon.sg", "Asia-Pacific", "SGD", "S$", ("singapore", "sg", "amazon.sg")),
    AmazonMarketplace("AE", "United Arab Emirates", "https://www.amazon.ae", "Middle East and North Africa", "AED", "AED", ("united arab emirates", "uae", "emirates", "ae", "amazon.ae")),
    AmazonMarketplace("SA", "Saudi Arabia", "https://www.amazon.sa", "Middle East and North Africa", "SAR", "SAR", ("saudi arabia", "saudi", "ksa", "sa", "amazon.sa")),
    AmazonMarketplace("EG", "Egypt", "https://www.amazon.eg", "Middle East and North Africa", "EGP", "E£", ("egypt", "eg", "amazon.eg")),
    AmazonMarketplace("ZA", "South Africa", "https://www.amazon.co.za", "Africa", "ZAR", "R", ("south africa", "za", "amazon.co.za")),
)

DEFAULT_MARKETPLACE_CODE = "US"
MARKETPLACE_BY_CODE = {marketplace.code: marketplace for marketplace in AMAZON_MARKETPLACES}
_ALIASES = {
    alias.lower(): marketplace.code
    for marketplace in AMAZON_MARKETPLACES
    for alias in marketplace.aliases
}


def normalize_marketplace_code(value: str | None) -> str:
    if not value:
        return DEFAULT_MARKETPLACE_CODE

    normalized = value.strip().lower()
    code = _ALIASES.get(normalized)
    if code:
        return code

    upper = normalized.upper()
    if upper in MARKETPLACE_BY_CODE:
        return upper
    return DEFAULT_MARKETPLACE_CODE


def marketplace_domain(code: str | None) -> str:
    return MARKETPLACE_BY_CODE[normalize_marketplace_code(code)].domain


def marketplace_currency(code: str | None) -> tuple[str, str]:
    marketplace = MARKETPLACE_BY_CODE[normalize_marketplace_code(code)]
    return marketplace.currency_code, marketplace.currency_symbol


def marketplace_label(code: str | None) -> str:
    marketplace = MARKETPLACE_BY_CODE[normalize_marketplace_code(code)]
    return f"{marketplace.country} ({marketplace.domain})"


# ---------------------------------------------------------------------------
# Rating node IDs for Amazon's p_72 URL filter parameter.
# These are marketplace-specific and control the "Avg. Customer Review" filter.
# ---------------------------------------------------------------------------

_RATING_NODES: dict[str, dict[int, str]] = {
    # amazon.com / amazon.in / amazon.com.* (most marketplaces share these)
    "US": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "IN": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "CA": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "MX": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "BR": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "AU": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "SG": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "AE": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "SA": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "EG": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "ZA": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    # European marketplaces
    "UK": {4: "328520031", 3: "328521031", 2: "328522031"},
    "DE": {4: "669342031", 3: "669343031", 2: "669344031"},
    "FR": {4: "1292115031", 3: "1292116031", 2: "1292117031"},
    "IT": {4: "1410764031", 3: "1410765031", 2: "1410766031"},
    "ES": {4: "1318568031", 3: "1318569031", 2: "1318570031"},
    "NL": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "SE": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "PL": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "TR": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "BE": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    "IE": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
    # Asia-Pacific
    "JP": {4: "2421889011", 3: "2421890011", 2: "2421891011"},
}


def amazon_rating_node_id(marketplace_code: str | None, min_rating: float) -> str | None:
    """
    Return the Amazon p_72 node ID for a given marketplace and minimum rating.

    Rounds the rating down to the nearest integer threshold (2, 3, or 4).
    Returns None if min_rating is below 2 (no practical filter) or above 5.
    """
    code = normalize_marketplace_code(marketplace_code)
    threshold = int(min_rating)            # 4.5 → 4, 3.2 → 3, 2.0 → 2
    threshold = max(2, min(threshold, 4))  # clamp to [2, 4]
    nodes = _RATING_NODES.get(code, _RATING_NODES["US"])
    return nodes.get(threshold)


def find_marketplace_in_text(message: str) -> str | None:
    lowered = message.lower()
    for alias in sorted(_ALIASES, key=len, reverse=True):
        code = _ALIASES[alias]
        if "." in alias and alias in lowered:
            return code
        if len(alias) <= 2:
            pattern = rf"\b(?:in|for|from|marketplace|region|country)\s+{re.escape(alias)}\b"
        else:
            pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, lowered, flags=re.I):
            return code
    return None


def strip_marketplace_terms(message: str) -> str:
    cleaned = message
    for marketplace in AMAZON_MARKETPLACES:
        aliases = sorted(marketplace.aliases, key=len, reverse=True)
        for alias in aliases:
            escaped = re.escape(alias)
            cleaned = re.sub(
                rf"\b(?:in|for|from|marketplace|region|country)\s+{escaped}\b",
                "",
                cleaned,
                flags=re.I,
            )
            cleaned = re.sub(rf"\bamazon\s+{escaped}\b", "", cleaned, flags=re.I)
            if "." in alias:
                cleaned = re.sub(rf"\b{escaped}\b", "", cleaned, flags=re.I)
    return " ".join(cleaned.split())
