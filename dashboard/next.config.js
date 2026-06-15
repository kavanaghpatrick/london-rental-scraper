/** @type {import('next').NextConfig} */

// Content-Security-Policy.
// Next.js (App Router) injects inline runtime scripts/styles, and Recharts
// applies inline styles, so 'unsafe-inline' is required for script/style.
// We still lock down framing, object embedding, base-uri, and form targets,
// and restrict connections/images to self + the Vercel Postgres host family.
const ContentSecurityPolicy = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self' https://*.vercel.app https://*.neon.tech",
  "frame-ancestors 'none'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  'upgrade-insecure-requests',
].join('; ');

const securityHeaders = [
  // Defense-in-depth content policy (see comment above).
  { key: 'Content-Security-Policy', value: ContentSecurityPolicy },
  // Block clickjacking (legacy header; CSP frame-ancestors is the modern equivalent).
  { key: 'X-Frame-Options', value: 'DENY' },
  // Stop MIME sniffing.
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  // Don't leak full URLs to third parties.
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  // Force HTTPS for 2 years, including subdomains (safe: site is HTTPS-only on Vercel).
  { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
  // Disable powerful browser features this app does not use.
  { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=(), browsing-topics=()' },
  // Isolate cross-origin resource loads.
  { key: 'X-DNS-Prefetch-Control', value: 'off' },
];

const nextConfig = {
  // Don't advertise the framework version in responses.
  poweredByHeader: false,

  async headers() {
    return [
      {
        // Apply to every route.
        source: '/:path*',
        headers: securityHeaders,
      },
    ];
  },
};

module.exports = nextConfig;
