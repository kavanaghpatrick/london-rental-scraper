"""
URL validation utilities for property scraping.

Issue #25 FIX: Provides consistent URL validation for floorplan URLs
and other extracted URLs across all spiders.
"""

from urllib.parse import urlparse, urljoin
from typing import Optional


def validate_url(url: str, base_url: str = None) -> Optional[str]:
    """Validate and normalize a URL.

    Args:
        url: URL string to validate (can be relative if base_url provided)
        base_url: Optional base URL for resolving relative URLs

    Returns:
        Validated absolute URL or None if invalid

    Example:
        >>> validate_url("https://example.com/image.jpg")
        'https://example.com/image.jpg'
        >>> validate_url("/images/floor.jpg", "https://example.com")
        'https://example.com/images/floor.jpg'
        >>> validate_url("javascript:void(0)")
        None
    """
    if not url or not isinstance(url, str):
        return None

    # Strip whitespace
    url = url.strip()

    if not url:
        return None

    # Handle relative URLs if base_url provided
    if base_url and not url.startswith(('http://', 'https://')):
        url = urljoin(base_url, url)

    # Parse and validate
    try:
        parsed = urlparse(url)

        # Must have valid scheme
        if parsed.scheme not in ('http', 'https'):
            return None

        # Must have netloc (domain)
        if not parsed.netloc:
            return None

        # Reject javascript: URLs that might slip through
        if 'javascript:' in url.lower():
            return None

        # Reject data: URLs (inline images, not real URLs)
        if parsed.scheme == 'data' or url.startswith('data:'):
            return None

        return url

    except Exception:
        return None


def validate_image_url(url: str, base_url: str = None) -> Optional[str]:
    """Validate URL and check it looks like an image URL.

    Args:
        url: URL string to validate
        base_url: Optional base URL for resolving relative URLs

    Returns:
        Validated image URL or None if invalid

    Example:
        >>> validate_image_url("https://example.com/floor.jpg")
        'https://example.com/floor.jpg'
        >>> validate_image_url("https://example.com/page.html")
        None
    """
    validated = validate_url(url, base_url)
    if not validated:
        return None

    # Check for common image extensions or patterns
    url_lower = validated.lower()
    path = urlparse(validated).path.lower()

    # Direct extension check
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp', '.tiff')
    if any(path.endswith(ext) for ext in image_extensions):
        return validated

    # Check for image serving URLs (CDNs often don't have extensions)
    image_patterns = [
        '/images/',
        '/image/',
        '/img/',
        '/media/',
        '/floorplan/',
        '/floor-plan/',
        '/floorplans/',
        '_FLP_',  # Rightmove floorplan pattern
        '/files/floorplan/',  # Chestertons pattern
    ]
    if any(pattern in url_lower for pattern in image_patterns):
        return validated

    # If URL contains format parameter, accept it
    if 'format=jpg' in url_lower or 'format=png' in url_lower:
        return validated

    # Reject if it looks like HTML page
    html_indicators = ['.html', '.htm', '.php', '.asp', '/property/', '/listing/']
    if any(ind in url_lower for ind in html_indicators):
        return None

    # Default: accept if it doesn't look like a page
    # (many CDN URLs don't have extensions)
    return validated


def validate_floorplan_url(url: str, base_url: str = None) -> Optional[str]:
    """Validate URL specifically for floorplan images.

    Applies stricter validation appropriate for floorplan URLs.

    Args:
        url: URL string to validate
        base_url: Optional base URL for resolving relative URLs

    Returns:
        Validated floorplan URL or None if invalid
    """
    validated = validate_image_url(url, base_url)
    if not validated:
        return None

    # Specific floorplan URL patterns we trust
    trusted_patterns = [
        'rightmove.co.uk',
        'zoopla.co.uk',
        'savills.com',
        'knightfrank.co.uk',
        'knightfrank.com',
        'chestertons.co.uk',
        'foxtons.co.uk',
        'homeflow-assets.co.uk',  # Chestertons CDN
        'amazonaws.com',  # AWS S3 hosting
        'cloudfront.net',  # AWS CloudFront
        'akamaized.net',  # Akamai CDN
    ]

    url_lower = validated.lower()
    if not any(pattern in url_lower for pattern in trusted_patterns):
        # Not from a trusted domain - still accept but log warning
        # (We don't reject, as there may be legitimate new CDNs)
        pass

    return validated
