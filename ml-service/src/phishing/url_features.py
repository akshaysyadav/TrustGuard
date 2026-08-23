import re
from urllib.parse import urlparse
import tldextract

def extract_18_url_features(raw_url: str) -> dict:
    """
    Extracts the 18 URL-derived lexical features matching PhiUSIIL dataset rules.
    """
    # 1. Normalize trailing slash for length compatibility
    clean_url = raw_url.rstrip('/')
    url_length = len(clean_url)

    # 2. Parse URL components
    parsed = urlparse(clean_url)
    ext = tldextract.extract(clean_url)

    domain_str = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    domain_length = len(domain_str)

    # 3. IP check
    is_domain_ip = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain_str) else 0

    # 4. TLD and Subdomain features
    tld = ext.suffix
    tld_length = len(tld)

    subdomain = ext.subdomain
    # Count subdomains excluding 'www'
    subdomains_list = [s for s in subdomain.split('.') if s and s != 'www']
    no_of_subdomain = len(subdomains_list)

    # 5. Domain-body letter count (matching PhiUSIIL SLD logic)
    sld_domain = ext.domain
    no_of_letters = sum(c.isalpha() for c in sld_domain)
    letter_ratio = no_of_letters / url_length if url_length > 0 else 0.0

    # 6. Digits in entire URL string
    no_of_digits = sum(c.isdigit() for c in clean_url)
    digit_ratio = no_of_digits / url_length if url_length > 0 else 0.0

    # 7. Character counts
    no_of_equals = clean_url.count('=')
    no_of_qmark = clean_url.count('?')
    no_of_ampersand = clean_url.count('&')

    # Standard special characters excluded from alphanumeric check
    alnum_or_standard_specials = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789=&#?.-:/_")
    no_of_other_specials = sum(1 for c in clean_url if c not in alnum_or_standard_specials)
    special_char_ratio = no_of_other_specials / url_length if url_length > 0 else 0.0

    # 8. HTTPS check
    is_https = 1 if clean_url.lower().startswith('https://') else 0

    # 9. Obfuscation features
    has_obfuscation = 1 if '%' in clean_url or '@' in clean_url else 0
    no_of_obfuscated_char = clean_url.count('%') + clean_url.count('@')
    obfuscation_ratio = no_of_obfuscated_char / url_length if url_length > 0 else 0.0

    return {
        'URLLength': url_length,
        'DomainLength': domain_length,
        'IsDomainIP': is_domain_ip,
        'TLDLength': tld_length,
        'NoOfSubDomain': no_of_subdomain,
        'HasObfuscation': has_obfuscation,
        'NoOfObfuscatedChar': no_of_obfuscated_char,
        'ObfuscationRatio': obfuscation_ratio,
        'NoOfLettersInURL': no_of_letters,
        'LetterRatioInURL': letter_ratio,
        'NoOfDegitsInURL': no_of_digits,
        'DegitRatioInURL': digit_ratio,
        'NoOfEqualsInURL': no_of_equals,
        'NoOfQMarkInURL': no_of_qmark,
        'NoOfAmpersandInURL': no_of_ampersand,
        'NoOfOtherSpecialCharsInURL': no_of_other_specials,
        'SpacialCharRatioInURL': special_char_ratio,
        'IsHTTPS': is_https
    }
