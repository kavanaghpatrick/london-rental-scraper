# Privacy Policy for Rent Fair Value Chrome Extension

**Last Updated:** January 17, 2026
**Extension Version:** 1.0.0

## Overview

Rent Fair Value ("the Extension") is a browser extension that provides machine learning-powered fair rent estimates for London rental properties. This privacy policy explains what data we collect, how we use it, and your rights regarding your data.

**Key Points:**
- The Extension operates anonymously - no login or account required
- All rent predictions are calculated locally in your browser
- We collect anonymous usage analytics to improve the Extension
- We do not sell your data to third parties

---

## 1. Data We Collect

### 1.1 Anonymous Usage Analytics

We use PostHog, a privacy-focused analytics platform, to understand how the Extension is used. When you use the Extension, we collect:

| Data Type | Example | Purpose |
|-----------|---------|---------|
| Page URL | `rightmove.co.uk/properties/123456` | Understand which property pages the Extension runs on |
| Site hostname | `rightmove.co.uk` | Track which supported sites are most used |
| Event type | `prediction_completed` | Measure feature usage and success rates |
| Property details | Bedrooms: 2, Postcode: SW3 | Improve prediction accuracy |
| Session identifier | `sess_abc123` | Group events within a browsing session |
| Extension version | `1.0.0` | Identify issues with specific versions |

**Events we track:**
- Extension loaded on a property page
- Prediction successfully generated
- Prediction failed (with error category)
- Floorplan OCR initiated/completed/failed
- Similar properties loaded
- Compare page opened

### 1.2 Data Stored Locally on Your Device

We store only one piece of data locally using Chrome's storage API:

| Data | Format | Purpose |
|------|--------|---------|
| Anonymous ID | `rfv_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` | Distinguish unique installations for analytics aggregation |

This identifier is:
- A randomly generated UUID (not linked to you personally)
- Never shared with property websites
- Stored only on your device
- Deletable by uninstalling the Extension

### 1.3 Property Data Processed

When you view a rental listing, the Extension extracts property information **directly from the page you're viewing**:

- Asking rent (price)
- Property address and postcode
- Number of bedrooms and bathrooms
- Property size (square footage)
- Property type (flat, house, etc.)
- Listed amenities
- Floorplan images (for OCR extraction)

**Important:** This property data is:
- Processed locally in your browser
- NOT stored by us
- NOT transmitted to our servers (except anonymized summaries in analytics)
- Only used to generate your rent estimate

---

## 2. How We Use Your Data

### 2.1 Analytics Data

We use anonymous analytics to:
- **Improve accuracy:** Understand prediction patterns across different property types
- **Fix bugs:** Identify and resolve errors affecting users
- **Measure performance:** Track OCR success rates and processing times
- **Guide development:** Prioritize features based on actual usage

### 2.2 Local Processing

All machine learning predictions are calculated **entirely within your browser** using a pre-trained XGBoost model. No property data is sent to external servers for prediction.

---

## 3. Third-Party Services

### 3.1 PostHog (Analytics)

- **Provider:** PostHog, Inc.
- **Data Center:** United States (us.i.posthog.com)
- **Data Sent:** Anonymous events with property metadata
- **Privacy Policy:** https://posthog.com/privacy
- **GDPR Compliance:** PostHog offers GDPR-compliant data processing

### 3.2 GitHub (Model Files)

- **Provider:** GitHub, Inc.
- **Purpose:** Host ML model and prediction cache files
- **Data Sent:** None (download only)
- **Privacy Policy:** https://docs.github.com/en/site-policy/privacy-policies/github-privacy-statement

### 3.3 Similar Properties API

- **Host:** Vercel-hosted API
- **Purpose:** Fetch comparable property listings
- **Data Sent:** Postcode, bedrooms, price range
- **Data Returned:** Anonymized similar listings (no personal data)

---

## 4. Data Retention

| Data Type | Retention Period |
|-----------|------------------|
| Analytics events | 90 days |
| Local anonymous ID | Until Extension uninstalled |
| Property data | Not retained (processed transiently) |

---

## 5. Your Rights (GDPR & UK GDPR)

As a user in the UK or EU, you have the right to:

### 5.1 Access Your Data
Contact us to receive a copy of any data associated with your anonymous identifier.

### 5.2 Delete Your Data
- **Local data:** Uninstall the Extension to delete all locally stored data
- **Analytics data:** Contact us to request deletion of analytics events associated with your anonymous ID

### 5.3 Opt Out of Analytics
We are working on an in-Extension toggle to disable analytics. Currently, you can:
- Use a content blocker to block `us.i.posthog.com`
- Contact us to add your anonymous ID to an exclusion list

### 5.4 Data Portability
Contact us to receive your data in a machine-readable format.

---

## 6. Data Security

We implement appropriate security measures:

- **Transport Security:** All data transmitted over HTTPS
- **No Sensitive Data:** We do not collect passwords, payment info, or personal identifiers
- **Minimal Data:** We collect only what's necessary for Extension functionality
- **Local Processing:** Predictions computed locally, reducing data exposure

---

## 7. Children's Privacy

The Extension is not directed at children under 13. We do not knowingly collect data from children.

---

## 8. Permissions Explained

The Extension requires the following Chrome permissions:

| Permission | Why We Need It |
|------------|----------------|
| `activeTab` | Read property details from the rental listing page you're viewing |
| `storage` | Store your anonymous identifier locally |
| Host permissions for property sites | Access property listing pages to extract data |
| Host permission for PostHog | Send anonymous analytics events |

---

## 9. Changes to This Policy

We may update this privacy policy from time to time. Changes will be reflected in the "Last Updated" date. Significant changes will be communicated via Extension update notes.

---

## 10. Contact Us

For privacy inquiries, data requests, or questions:

- **Email:** privacy@rentfairvalue.com
- **GitHub Issues:** https://github.com/kavanaghpatrick/london-rental-scraper/issues

---

## 11. Legal Basis for Processing (GDPR)

Our legal basis for processing analytics data is **legitimate interest** in improving the Extension. We have assessed that this processing:
- Is necessary for our legitimate business purposes
- Does not override your privacy rights (data is anonymous)
- Uses privacy-preserving techniques (random identifiers, aggregation)

---

## Summary

| Question | Answer |
|----------|--------|
| Do you collect personal information? | No |
| Do you require login/account? | No |
| Can you identify me personally? | No |
| Where are predictions calculated? | Locally in your browser |
| Do you sell data? | No |
| Can I delete my data? | Yes, uninstall the Extension |
