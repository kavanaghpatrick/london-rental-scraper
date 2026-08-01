/**
 * FOR-SALE XGBoost Predictor — the SALE analogue of chrome-extension/xgboost.js, ported
 * BY VALUE (NOT imported) so the sale module is a self-contained, vendorable file with
 * ZERO rental coupling. Shared by the Chrome extension (browser For-Sale mode) AND
 * serving's Next.js /api/predict-sale route (Node). Pure / Node-safe: no window/document.
 *
 * ISOLATION (Inc4 RENTAL-REGRESSION LOCK):
 *   - SEPARATE model cache global: globalThis.__SALE_XGB_MODEL_CACHE__ (never the rental
 *     __XGB_MODEL_CACHE__), so the rental offline test can never collide with the sale one.
 *   - SEPARATE artifacts: it consumes the Inc3 Booster JSON (output/sale_api/model.json),
 *     features.json (the 34-name order), and the baked district_freq / postcode_area_freq
 *     maps from output/sale_model_inference.json (BAKED into this file as module constants).
 *   - chrome-extension/xgboost.js stays byte-identical; this file imports nothing.
 *
 * It mirrors for_sale/sale_price_model.build_features(inference=True) + for_sale/
 * sale_features.py exactly, emitting the 34 FEATURE_COLUMNS in frozen order. Math.fround is
 * applied on BOTH sides of every split (the sale model has monotone_constraints, so a
 * float64 mis-branch can surface as a visible non-monotone dip and exceed TOL=0.005).
 */

// Module-scope cache of parsed {model, features}, SALE-namespaced (separate from the rental
// __XGB_MODEL_CACHE__). On a warm serverless invocation this reuses the already-parsed
// Booster instead of re-fetching/re-parsing it.
const _SALE_XGB_MODEL_CACHE = (typeof globalThis !== 'undefined'
  ? (globalThis.__SALE_XGB_MODEL_CACHE__ || (globalThis.__SALE_XGB_MODEL_CACHE__ = new Map()))
  : new Map());

class SaleXGBoostPredictor {
  constructor() {
    this.model = null;
    this.features = null;
    this.loaded = false;
  }

  async load(modelUrl, featuresUrl) {
    if (this.loaded) return;

    // Reuse a module-scope-cached parse if this URL pair was loaded before.
    const cacheKey = `${modelUrl}|${featuresUrl}`;
    const cached = _SALE_XGB_MODEL_CACHE.get(cacheKey);
    if (cached) {
      this.model = cached.model;
      this.features = cached.features;
      this.loaded = true;
      return;
    }

    const [modelRes, featuresRes] = await Promise.all([
      fetch(modelUrl),
      fetch(featuresUrl)
    ]);

    this.model = await modelRes.json();
    this.features = await featuresRes.json();
    this.loaded = true;
    _SALE_XGB_MODEL_CACHE.set(cacheKey, { model: this.model, features: this.features });
  }

  predict(featureDict) {
    if (!this.loaded) {
      throw new Error('Sale model not loaded');
    }

    // Build feature array in the trained order; any absent key -> 0.
    const featureArray = this.features.map(name => featureDict[name] ?? 0);

    // base_score can be: number, array, or string like "[8.399085E0]".
    let baseScoreRaw = this.model.learner.learner_model_param.base_score;
    if (typeof baseScoreRaw === 'string') {
      baseScoreRaw = baseScoreRaw.replace(/[\[\]]/g, '');
    }
    const baseScore = parseFloat(Array.isArray(baseScoreRaw) ? baseScoreRaw[0] : baseScoreRaw);

    const trees = this.model.learner.gradient_booster.model.trees;
    let sum = baseScore;
    for (const tree of trees) {
      sum += this.predictTree(tree, featureArray);
    }
    return sum; // LOG value; caller applies Math.expm1
  }

  predictTree(tree, features) {
    let nodeId = 0;
    while (true) {
      const leftChildren = tree.left_children[nodeId];
      const rightChildren = tree.right_children[nodeId];

      // Leaf node: the prediction value lives in split_conditions, NOT base_weights.
      if (leftChildren === -1) {
        return parseFloat(tree.split_conditions[nodeId]);
      }

      const splitIndex = tree.split_indices[nodeId];
      // XGBoost evaluates the split in float32; Math.fround BOTH sides so the comparison
      // matches native XGBoost (load-bearing — float64 mis-branches exceed TOL over the
      // monotone-constrained ensemble).
      const splitCondition = Math.fround(parseFloat(tree.split_conditions[nodeId]));
      const featureValue = features[splitIndex] ?? 0;

      const defaultLeft = tree.default_left ? tree.default_left[nodeId] : true;

      if (featureValue === null || featureValue === undefined || Number.isNaN(featureValue)) {
        nodeId = defaultLeft ? leftChildren : rightChildren;
      } else if (Math.fround(featureValue) < splitCondition) {
        nodeId = leftChildren;
      } else {
        nodeId = rightChildren;
      }
    }
  }
}

// ── Sale feature engineering — ports build_features(inference=True) + sale_features.py ───
const SaleXGBFeatures = {
  // Prime districts (mirror sale_price_model.PRIME_POSTCODES, re-declared not imported).
  PRIME_POSTCODES: ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8'],

  // Prestige street -> tier (mirror sale_price_model.PRESTIGE_STREET_TIER verbatim).
  PRESTIGE_STREET_TIER: {
    'eaton square': 4, 'belgrave square': 4, 'the boltons': 4, 'wilton crescent': 4,
    'cadogan square': 3, 'lowndes square': 3, 'cheyne walk': 3, 'hans place': 3,
    'sloane street': 3, 'chester square': 3, 'grosvenor square': 4,
    'onslow square': 2, 'lennox gardens': 2, 'tregunter road': 2, 'pont street': 2,
    'thurloe square': 2, 'egerton gardens': 2, 'draycott place': 2, 'carlyle square': 2,
    'holland park': 1, 'elm park gardens': 1, 'phillimore gardens': 1,
  },

  // Baked freq maps + numeric defaults — verbatim from output/sale_model_inference.json
  // (the BLOCKER-1 single-row degeneracy fix). The parity gate proves these match Python.
  DISTRICT_FREQ: {
    'UNKNOWN': 0.1580407538505692, 'SW6': 0.09432698746771262, 'NW3': 0.07701138429159093,
    'NW8': 0.0667750884913422, 'SW3': 0.05673012532287382, 'SW7': 0.05567779584808189,
    'W8': 0.040275518989763705, 'SW10': 0.03922318951497178, 'W2': 0.03864919161963073,
    'W11': 0.037214196881278104, 'SW5': 0.03176121687553812, 'SW1W': 0.028699894767052522,
    'SW1X': 0.026499569501578495, 'W14': 0.025734238974457094, 'NW1': 0.024203577920214294,
    'W10': 0.01884626423036449, 'W1K': 0.017506935807902038, 'W1H': 0.01683727159667081,
    'SW11': 0.015210944226537836, 'W1U': 0.014254281067636085, 'SW18': 0.013488950540514685,
    'W1J': 0.011862623170381708, 'W1W': 0.007366306323543481, 'W6': 0.007366306323543481,
    'SW15': 0.00679230842820243, 'SW1V': 0.006122644216971204, 'SW19': 0.005739978953410504,
    'W1G': 0.005739978953410504, 'SW1P': 0.005548646321630154, 'SW13': 0.005070314742179278,
    'NW11': 0.004400650530948053, 'W12': 0.0036353200038266526, 'NW2': 0.0033483210561561275,
    'W1S': 0.003061322108485602, 'W1B': 0.002965655792595427, 'W9': 0.002869989476705252,
    'SW12': 0.002487324213144552, 'NW6': 0.002391657897254377, 'SW17': 0.002295991581364202,
    'W1': 0.002295991581364202, 'SW14': 0.0021046589495838514, 'W1T': 0.0018176600019133263,
    'SW1E': 0.001434994738352626, 'SW1': 0.001339328422462451, 'SW1H': 0.001147995790682101,
    'W13': 0.0008609968430115756, 'SW1A': 0.0005739978953410505, 'W1F': 0.0005739978953410505,
    'SW16': 0.00038266526356070025, 'W1D': 0.00028699894767052523, 'N2': 0.00028699894767052523,
    'SW1Y': 0.00019133263178035013, 'WC2H': 9.566631589017506e-05, 'N3': 9.566631589017506e-05,
    'N6': 9.566631589017506e-05, 'EC2A': 9.566631589017506e-05, 'NW5': 9.566631589017506e-05,
    'BH21': 9.566631589017506e-05, 'TW7': 9.566631589017506e-05,
  },
  DISTRICT_FREQ_DEFAULT: 9.566631589017506e-05,
  POSTCODE_AREA_FREQ: {
    'SW': 0.40285085621352723, 'W': 0.2600210465894958, 'NW': 0.17822634650339617,
    'UNKNOWN': 0.1580407538505692, 'N': 0.00047833157945087533, 'WC': 9.566631589017506e-05,
    'EC': 9.566631589017506e-05, 'BH': 9.566631589017506e-05, 'TW': 9.566631589017506e-05,
  },
  POSTCODE_AREA_FREQ_DEFAULT: 9.566631589017506e-05,

  // Distance constants (mirror for_sale/sale_features.py). DEFAULT_CENTER_DISTANCE_KM is the
  // NEUTRAL coordless fill (NOT 0 — the centroid is out-of-distribution).
  DEFAULT_CENTER_DISTANCE_KM: 3.3892584524370477,
  SALE_CITY_CENTER: { lat: 51.5074, lon: -0.1278 },
  EARTH_RADIUS_KM: 6371.0088,

  // Property-type vocab (mirror sale_price_model._HOUSE_TYPES / _FLAT_TYPES).
  HOUSE_TYPES: ['house', 'town house', 'townhouse', 'terraced', 'detached', 'semi'],
  FLAT_TYPES: ['flat', 'apartment', 'maisonette', 'penthouse', 'studio'],

  // Boundary-anchored outward-code district (mirror sale_price_model.postcode_to_district).
  _postcodeDistrict(postcode) {
    if (postcode === null || postcode === undefined || postcode === '') return 'UNKNOWN';
    const s = String(postcode).trim().toUpperCase();
    const m = s.match(/^([A-Z]{1,2}\d{1,2}[A-Z]?)(?=\s|\d|$)/);
    return m ? m[1] : 'UNKNOWN';
  },

  // Alpha-area (mirror sale_price_model.postcode_to_area).
  _postcodeArea(postcode) {
    if (postcode === null || postcode === undefined || postcode === '') return 'UNKNOWN';
    const s = String(postcode).trim().toUpperCase();
    const m = s.match(/^([A-Z]{1,2})/);
    return m ? m[1] : 'UNKNOWN';
  },

  // Prestige tier (mirror sale_price_model._prestige_tier).
  _prestigeTier(address) {
    if (!address) return 0;
    const al = String(address).toLowerCase();
    let best = 0;
    for (const [street, tier] of Object.entries(this.PRESTIGE_STREET_TIER)) {
      if (al.includes(street) && tier > best) best = tier;
    }
    return best;
  },

  // Baseline house/flat (mirror sale_price_model._classify_type). A plain Flat/Apartment
  // (and anything matching neither) defaults is_flat=1.
  _classifyType(propertyType) {
    const pt = (propertyType === null || propertyType === undefined ? '' : String(propertyType)).toLowerCase();
    let isHouse = this.HOUSE_TYPES.some(t => pt.includes(t)) ? 1 : 0;
    let isFlat = this.FLAT_TYPES.some(t => pt.includes(t)) ? 1 : 0;
    if (!isHouse && !isFlat) isFlat = 1;
    return [isHouse, isFlat];
  },

  // Extended one-hots (mirror sale_features._classify_type_extended).
  _classifyTypeExtended(propertyType) {
    const pt = (propertyType === null || propertyType === undefined ? '' : String(propertyType)).toLowerCase();
    const isPenthouse = pt.includes('penthouse') ? 1 : 0;
    const isMaisonette = (pt.includes('maisonette') || pt.includes('duplex')) ? 1 : 0;
    const isTerraced = (pt.includes('terraced') || pt.includes('town house') ||
      pt.includes('townhouse') || pt.includes('end of terrace')) ? 1 : 0;
    const isStudio = pt.includes('studio') ? 1 : 0;
    return [isPenthouse, isMaisonette, isTerraced, isStudio];
  },

  // Great-circle km to SALE_CITY_CENTER (mirror sale_features._haversine_km).
  _haversineKm(lat, lon) {
    const toRad = x => x * Math.PI / 180;
    const lat1 = toRad(lat);
    const lon1 = toRad(lon);
    const lat2 = toRad(this.SALE_CITY_CENTER.lat);
    const lon2 = toRad(this.SALE_CITY_CENTER.lon);
    const dlat = lat1 - lat2;
    const dlon = lon1 - lon2;
    let a = Math.sin(dlat / 2.0) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2.0) ** 2;
    a = Math.min(Math.max(a, 0.0), 1.0); // clip to [0,1]
    return 2.0 * this.EARTH_RADIUS_KM * Math.asin(Math.sqrt(a));
  },

  _districtFreq(district) {
    return Object.prototype.hasOwnProperty.call(this.DISTRICT_FREQ, district)
      ? this.DISTRICT_FREQ[district]
      : this.DISTRICT_FREQ_DEFAULT;
  },

  _postcodeAreaFreq(area) {
    return Object.prototype.hasOwnProperty.call(this.POSTCODE_AREA_FREQ, area)
      ? this.POSTCODE_AREA_FREQ[area]
      : this.POSTCODE_AREA_FREQ_DEFAULT;
  },

  // Coerce a possibly-string numeric to a finite number, else NaN (pandas to_numeric).
  _num(v) {
    if (v === null || v === undefined || v === '') return NaN;
    const n = Number(v);
    return Number.isNaN(n) ? NaN : n;
  },

  // Emit exactly the 34 FEATURE_COLUMNS for a single property dict — mirrors
  // build_features(inference=True) BY VALUE (single-row branch).
  buildFeatures(data) {
    data = data || {};

    // Core numerics with fillna(1) — only NaN/absent maps to 1; a literal 0 stays 0.
    let bedrooms = this._num(data.bedrooms);
    bedrooms = Number.isNaN(bedrooms) ? 1.0 : bedrooms;
    let bathrooms = this._num(data.bathrooms);
    bathrooms = Number.isNaN(bathrooms) ? 1.0 : bathrooms;

    // size_sqft single-row branch: NaN/absent -> 700.0; a literal 0 STAYS 0 (AMENDMENT FIX 4).
    let size = this._num(data.size_sqft);
    if (Number.isNaN(size)) size = 700.0;

    // beds_adj = bedrooms with 0 replaced by 0.5.
    const bedsAdj = bedrooms === 0 ? 0.5 : bedrooms;

    // Size family.
    const logSqft = Math.log1p(size);
    const sqrtSqft = Math.sqrt(size);
    const sizePerBed = size / bedsAdj;
    const bedsSquared = bedrooms ** 2;
    const sizeSquared = (size ** 2) / 100000;
    const isTiny = size < 400 ? 1 : 0;
    const isHuge = size >= 3000 ? 1 : 0;

    // Bathroom family.
    const bathRatio = bathrooms / bedsAdj;
    const excessBathrooms = Math.max(0, bathrooms - bedrooms);
    const bedBathInteraction = bedrooms * bathrooms;
    const hasEnsuiteEach = bathRatio >= 1 ? 1 : 0;
    const highBathroomCount = bathrooms >= 4 ? 1 : 0;

    // Location family.
    const district = this._postcodeDistrict(data.postcode);
    const area = this._postcodeArea(data.postcode);
    const isPrimePostcode = this.PRIME_POSTCODES.some(p => district.startsWith(p)) ? 1 : 0;
    const prestigeTier = this._prestigeTier(data.address);
    const districtFreq = this._districtFreq(district);
    const postcodeAreaFreq = this._postcodeAreaFreq(area);

    // Property type one-hots.
    const [isHouse, isFlat] = this._classifyType(data.property_type);
    const [isPenthouse, isMaisonette, isTerraced, isStudio] = this._classifyTypeExtended(data.property_type);

    // is_new_build — fillna(0); only present-and-numeric is used.
    let isNewBuild = this._num(data.is_new_build);
    isNewBuild = Number.isNaN(isNewBuild) ? 0 : Math.trunc(isNewBuild);

    // Distance family. Coordless (lat OR lon absent/NaN) -> neutral DEFAULT; else haversine,
    // NaN -> DEFAULT. Coerce BEFORE log1p.
    const latRaw = this._num(data.latitude);
    const lonRaw = this._num(data.longitude);
    const hasLat = (data.latitude !== null && data.latitude !== undefined) && !Number.isNaN(latRaw);
    const hasLon = (data.longitude !== null && data.longitude !== undefined) && !Number.isNaN(lonRaw);
    let centerDistanceKm;
    if (!hasLat || !hasLon) {
      centerDistanceKm = this.DEFAULT_CENTER_DISTANCE_KM;
    } else {
      const km = this._haversineKm(latRaw, lonRaw);
      centerDistanceKm = Number.isNaN(km) ? this.DEFAULT_CENTER_DISTANCE_KM : km;
    }
    const logCenterDistance = Math.log1p(centerDistanceKm);
    const centerDistanceInv = 1.0 / (1.0 + centerDistanceKm);

    // Interactions (magnitude-free shapes).
    const sizePrimeInteraction = size * isPrimePostcode;
    const sizeXCentral = size * centerDistanceInv;
    const houseSizeInteraction = isHouse * size;
    const prestigeTierXSize = prestigeTier * size;

    // Sale-only qualifier (from the price_qualifier STRING only).
    const q = data.price_qualifier === null || data.price_qualifier === undefined ? '' : String(data.price_qualifier);
    const priceQualifierPoa = q.toUpperCase().startsWith('POA') ? 1 : 0;

    // Return all 34 keys in the frozen FEATURE_COLUMNS order.
    return {
      bedrooms: bedrooms,
      bathrooms: bathrooms,
      size_sqft: size,
      log_sqft: logSqft,
      sqrt_sqft: sqrtSqft,
      size_per_bed: sizePerBed,
      beds_squared: bedsSquared,
      size_squared: sizeSquared,
      bath_ratio: bathRatio,
      excess_bathrooms: excessBathrooms,
      bed_bath_interaction: bedBathInteraction,
      is_prime_postcode: isPrimePostcode,
      prestige_tier: prestigeTier,
      district_freq: districtFreq,
      is_house: isHouse,
      is_flat: isFlat,
      is_new_build: isNewBuild,
      size_prime_interaction: sizePrimeInteraction,
      is_tiny: isTiny,
      is_huge: isHuge,
      has_ensuite_each: hasEnsuiteEach,
      high_bathroom_count: highBathroomCount,
      postcode_area_freq: postcodeAreaFreq,
      is_penthouse: isPenthouse,
      is_maisonette: isMaisonette,
      is_terraced: isTerraced,
      is_studio: isStudio,
      center_distance_km: centerDistanceKm,
      log_center_distance: logCenterDistance,
      center_distance_inv: centerDistanceInv,
      size_x_central: sizeXCentral,
      house_size_interaction: houseSizeInteraction,
      prestige_tier_x_size: prestigeTierXSize,
      price_qualifier_poa: priceQualifierPoa,
    };
  },
};

// Universal export: extension content-script globals AND Node/CommonJS (the /api/predict-sale
// route imports the SAME source, so client + server feature builders can never drift).
if (typeof window !== 'undefined') {
  window.SaleXGBoostPredictor = SaleXGBoostPredictor;
  window.SaleXGBFeatures = SaleXGBFeatures;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { SaleXGBoostPredictor, SaleXGBFeatures };
}
