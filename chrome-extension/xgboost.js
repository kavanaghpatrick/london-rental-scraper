/**
 * XGBoost Predictor for Browser
 * Loads exported XGBoost JSON model and runs predictions client-side
 */

class XGBoostPredictor {
  constructor() {
    this.model = null;
    this.features = null;
    this.loaded = false;
  }

  async load(modelUrl, featuresUrl) {
    if (this.loaded) return;

    console.log('[XGB] Loading model...');
    const [modelRes, featuresRes] = await Promise.all([
      fetch(modelUrl),
      fetch(featuresUrl)
    ]);

    this.model = await modelRes.json();
    this.features = await featuresRes.json();
    this.loaded = true;

    // base_score can be: number, array, or string like "[8.399085E0]"
    let baseScoreRaw = this.model.learner.learner_model_param.base_score;
    if (typeof baseScoreRaw === 'string') {
      baseScoreRaw = baseScoreRaw.replace(/[\[\]]/g, ''); // Remove brackets
    }
    const baseScore = parseFloat(Array.isArray(baseScoreRaw) ? baseScoreRaw[0] : baseScoreRaw);
    console.log(`[XGB] Loaded model with ${this.model.learner.gradient_booster.model.trees.length} trees, ${this.features.length} features, base_score=${baseScore}`);
  }

  predict(featureDict) {
    if (!this.loaded) {
      throw new Error('Model not loaded');
    }

    // Build feature array in correct order
    const featureArray = this.features.map(name => featureDict[name] ?? 0);

    // Get base score (can be number, array, or string like "[8.399085E0]")
    let baseScoreRaw = this.model.learner.learner_model_param.base_score;
    if (typeof baseScoreRaw === 'string') {
      baseScoreRaw = baseScoreRaw.replace(/[\[\]]/g, '');
    }
    const baseScore = parseFloat(Array.isArray(baseScoreRaw) ? baseScoreRaw[0] : baseScoreRaw);

    // Sum predictions from all trees
    const trees = this.model.learner.gradient_booster.model.trees;
    let sum = baseScore;

    console.log(`[XGB] Predicting with base_score=${baseScore}, ${trees.length} trees`);
    console.log(`[XGB] First 5 features:`, featureArray.slice(0, 5));

    for (const tree of trees) {
      sum += this.predictTree(tree, featureArray);
    }

    console.log(`[XGB] Final prediction (log): ${sum}`);
    return sum;
  }

  predictTree(tree, features) {
    // Navigate tree from root (node 0)
    let nodeId = 0;

    while (true) {
      const leftChildren = tree.left_children[nodeId];
      const rightChildren = tree.right_children[nodeId];

      // Leaf node (no children)
      // IMPORTANT: For leaf nodes in XGBoost JSON format, the prediction value is stored
      // in split_conditions, NOT base_weights. base_weights contains gradient information
      // from training, while split_conditions[leaf_node] holds the actual leaf value.
      if (leftChildren === -1) {
        return parseFloat(tree.split_conditions[nodeId]);
      }

      const splitIndex = tree.split_indices[nodeId];
      const splitCondition = parseFloat(tree.split_conditions[nodeId]);
      const featureValue = features[splitIndex] ?? 0;

      // XGBoost: left if value < condition, right otherwise
      // Handle missing values (default direction)
      const defaultLeft = tree.default_left ? tree.default_left[nodeId] : true;

      if (featureValue === null || featureValue === undefined || Number.isNaN(featureValue)) {
        nodeId = defaultLeft ? leftChildren : rightChildren;
      } else if (featureValue < splitCondition) {
        nodeId = leftChildren;
      } else {
        nodeId = rightChildren;
      }
    }
  }
}

// Feature engineering helpers
const XGBFeatures = {
  PRIME_POSTCODES: ['SW1', 'SW3', 'SW7', 'SW10', 'W1', 'W8', 'W11', 'NW3', 'NW8'],
  CITY_CENTER: { lat: 51.5074, lon: -0.1278 },

  // === V16/V20 SOCIAL HOUSING DETECTION ===
  // Social housing estates (from V16 investigation)
  SOCIAL_ESTATE_PATTERNS: [
    /world'?s?\s*end\s*estate/i, /townshend\s*estate/i, /hallfield\s*estate/i,
    /churchill\s*gardens/i, /ebury\s*bridge/i, /peabody/i,
    /trellick\s*tower/i, /lancaster\s*west/i, /silchester/i,
    /lisson\s*grove/i, /penfold\s*place/i, /mallory\s*street/i
  ],

  // Premium postcodes for social housing detection
  PREMIUM_POSTCODES_SOCIAL: ['SW1', 'SW3', 'SW7', 'W1', 'W8', 'NW8'],

  // Postcode adjustment factors (from V16 investigation)
  POSTCODE_ADJUSTMENTS: {
    'W1J': 1.41, 'W1S': 1.47, 'W1K': 1.25, 'W8': 1.20, 'SW1X': 1.13,
    'SW14': 0.83, 'SW17': 0.65, 'SW18': 0.80, 'SW12': 0.64, 'SW13': 0.90,
    'default': 1.0
  },

  // === V16 ADDRESS-BASED PREMIUM FEATURES ===
  // Prestigious garden squares - command massive premiums
  GARDEN_SQUARES: [
    'cadogan square', 'belgrave square', 'chester square', 'eaton square',
    'montpelier square', 'brompton square', 'thurloe square', 'lowndes square',
    'trevor square', 'lennox gardens', 'cadogan gardens', 'sloane square',
    'paultons square', 'chelsea square', 'onslow square', 'pelham crescent',
    'egerton crescent', 'egerton gardens', 'ovington square'
  ],

  // Ultra-prime addresses - highest tier
  ULTRA_PRIME_ADDRESSES: [
    'belgrave square', 'chester square', 'eaton square', 'wilton crescent',
    'grosvenor square', 'grosvenor crescent', 'upper grosvenor street',
    'park lane', 'hamilton terrace', 'avenue road', 'bishops avenue'
  ],

  // Prime streets/places - high tier
  PRIME_STREETS: [
    'cadogan square', 'cadogan place', 'cadogan gardens', 'hans place',
    'lennox gardens', 'pont street', 'sloane street', 'draycott place',
    'draycott avenue', 'eaton place', 'eaton terrace', 'montpelier street',
    'brompton square', 'thurloe square', 'ennismore gardens', 'princes gate',
    'hyde park gate', 'kensington palace gardens', 'palace gardens terrace',
    'campden hill', 'holland park', 'phillimore gardens', 'carlyle square',
    'cheyne walk', 'the boltons', 'tregunter road', 'elm park gardens'
  ],

  // === V20+ PRESTIGE LOCATION PPSF (from data analysis) ===
  // Much more accurate than binary is_garden_square
  PRESTIGE_LOCATION_PPSF: {
    'draycott avenue': 18.28,
    'trevor square': 13.24,
    'hyde park gate': 12.88,
    'wilton crescent': 11.23,
    'cadogan square': 10.99,
    'lowndes square': 9.11,
    'thurloe square': 8.14,
    'pont street': 7.92,
    'princes gate': 7.85,
    'ennismore gardens': 7.54,
    'cadogan place': 6.93,
    'montpelier square': 6.91,
    'cadogan gardens': 6.78,
    'lennox gardens': 6.76,
    'eaton square': 6.72,
    'hans place': 6.60,
    'sloane square': 6.26,
    'onslow square': 6.19,
    'palace gardens terrace': 6.11,
    'chester square': 6.08,
    'sloane street': 6.07,
    'cheyne walk': 6.07,
    'egerton gardens': 5.92,
    'the boltons': 5.92,
    'brompton square': 5.47,
    'draycott place': 5.43,
    'ovington square': 4.77,
    'holland park': 4.63,
    'elm park gardens': 4.48,
    'belgrave square': 6.76,
    'default': 5.42,
  },

  // V20+ Prestige tier classification (0-4)
  PRESTIGE_LOCATION_TIER: {
    // Ultra tier (>80% premium): 4
    'trevor square': 4, 'cadogan square': 4, 'wilton crescent': 4,
    'hyde park gate': 4, 'draycott avenue': 4,
    // High tier (40-80% premium): 3
    'lowndes square': 3, 'thurloe square': 3, 'pont street': 3,
    'princes gate': 3, 'cadogan place': 3, 'hans place': 3,
    'sloane street': 3, 'the boltons': 3,
    // Premium tier (20-40% premium): 2
    'ennismore gardens': 2, 'montpelier square': 2, 'cadogan gardens': 2,
    'lennox gardens': 2, 'eaton square': 2, 'belgrave square': 2,
    'sloane square': 2, 'palace gardens terrace': 2, 'chester square': 2,
    'cheyne walk': 2, 'draycott place': 2,
    // Standard tier (<20% premium): 1
    'onslow square': 1, 'egerton gardens': 1, 'brompton square': 1,
    'ovington square': 1, 'holland park': 1, 'elm park gardens': 1,
  },

  // V20+ Tier multipliers (for addresses not in specific lookup)
  PRESTIGE_TIER_MULTIPLIER: {
    4: 2.03,  // Ultra: +103% premium
    3: 1.33,  // High: +33% premium
    2: 1.13,  // Premium: +13% premium
    1: 1.01,  // Standard: +1% premium
    0: 1.00,  // None
  },

  // Tube stations matching Python model (16 stations for accurate distance calculations)
  TUBE_STATIONS: {
    'South Kensington': { lat: 51.4941, lon: -0.1738 },
    'Sloane Square': { lat: 51.4924, lon: -0.1565 },
    'Knightsbridge': { lat: 51.5015, lon: -0.1607 },
    'Hyde Park Corner': { lat: 51.5027, lon: -0.1527 },
    'Green Park': { lat: 51.5067, lon: -0.1428 },
    'Bond Street': { lat: 51.5142, lon: -0.1494 },
    'Notting Hill Gate': { lat: 51.5094, lon: -0.1967 },
    'High Street Kensington': { lat: 51.5009, lon: -0.1925 },
    'Earls Court': { lat: 51.4914, lon: -0.1934 },
    'Gloucester Road': { lat: 51.4945, lon: -0.1829 },
    'St Johns Wood': { lat: 51.5347, lon: -0.1740 },
    'Hampstead': { lat: 51.5566, lon: -0.1780 },
    'Baker Street': { lat: 51.5226, lon: -0.1571 },
    'Victoria': { lat: 51.4965, lon: -0.1447 },
    'Westminster': { lat: 51.5014, lon: -0.1248 },
    'Paddington': { lat: 51.5154, lon: -0.1755 },
    'Canary Wharf': { lat: 51.5033, lon: -0.0181 },
  },

  // Postcode one-hot features (matching features.json)
  POSTCODE_FEATURES: ['pc_SW3', 'pc_SW7', 'pc_W8', 'pc_W2', 'pc_SW5', 'pc_SW11', 'pc_SW10',
                      'pc_NW8', 'pc_W11', 'pc_SW1X', 'pc_NW3', 'pc_SW1W', 'pc_W14', 'pc_NW1', 'pc_W10'],

  // Property type one-hot features (matching features.json V20 model)
  PROPERTY_TYPE_FEATURES: ['type_apartment', 'type_detached', 'type_duplex',
                           'type_end of terrace', 'type_flat', 'type_ground flat',
                           'type_house', 'type_house boat', 'type_house of multiple occupation',
                           'type_house share', 'type_link detached house', 'type_long let',
                           'type_maisonette', 'type_mews', 'type_not specified', 'type_parking',
                           'type_penthouse', 'type_semi-detached', 'type_studio',
                           'type_terraced', 'type_town house'],

  // Premium agents list (matching Python PREMIUM_AGENTS + Carter Jonas)
  PREMIUM_AGENTS: ['knight frank', 'savills', 'harrods', 'sotheby', 'beauchamp', 'strutt', 'chestertons', 'carter jonas', 'hamptons', 'winkworth', 'marsh'],

  // === V17 PROPERTY TYPE CATEGORIES ===
  // V19 UPDATE: Mews are now separate from houses (they have distinct pricing)
  HOUSE_TYPES: ['house', 'terraced', 'detached', 'semi-detached', 'town house', 'cottage', 'end of terrace', 'link detached'],
  FLAT_TYPES: ['flat', 'apartment', 'studio', 'penthouse', 'maisonette', 'duplex', 'ground flat'],

  // === V17 TARGET-ENCODED PPSF VALUES (from training data) ===
  OVERALL_HOUSE_PPSF: 5.5635,  // Median PPSF for houses
  OVERALL_FLAT_PPSF: 5.2459,   // Median PPSF for flats
  OVERALL_PPSF: 5.2093,        // Overall median PPSF
  // V19: Mews-specific PPSF (distinct from houses)
  OVERALL_MEWS_PPSF: 5.40,     // Median PPSF for mews properties

  // Postcode × Property Type PPSF lookup (house PPSF by postcode district)
  PC_HOUSE_PPSF: {
    'SW3': 5.46, 'SW7': 6.02, 'SW1X': 8.28, 'NW8': 5.76, 'SW10': 6.26,
    'NW3': 4.82, 'SW6': 3.52, 'W8': 5.93, 'W11': 6.32, 'SW1W': 6.96,
    'W6': 6.75, 'W2': 5.17, 'W14': 5.33, 'NW1': 5.76, 'SW18': 3.42,
    'SW5': 5.50, 'SW11': 5.00, 'W10': 5.00, // defaults for missing
    'default': 5.56  // overall house median
  },
  PC_FLAT_PPSF: {
    'SW3': 6.50, 'SW7': 6.20, 'SW1X': 7.80, 'NW8': 5.50, 'SW10': 5.80,
    'NW3': 5.20, 'SW6': 4.50, 'W8': 6.80, 'W11': 6.00, 'SW1W': 6.50,
    'W6': 5.50, 'W2': 5.80, 'W14': 5.20, 'NW1': 5.50, 'SW5': 5.00,
    'SW11': 4.80, 'W10': 4.50,
    'default': 5.25  // overall flat median
  },

  // V19: Mews-specific PPSF by postcode (mews have distinct pricing from houses)
  // Note: SW1X is skewed by one ultra-prime outlier; SW1W is more typical Belgravia mews
  PC_MEWS_PPSF: {
    'SW1X': 7.00, 'SW1W': 4.36, 'SW1V': 4.51, 'SW1': 5.50,  // SW1 sub-districts
    'SW10': 8.28, 'SW3': 5.62, 'SW7': 4.14, 'SW5': 3.67, 'SW11': 3.22,
    'W1H': 8.33, 'W1G': 5.06, 'W8': 4.51, 'W2': 4.40, 'W11': 4.11, 'W12': 3.98,
    'NW3': 4.92,
    'default': 5.40  // overall mews median
  },

  // Source quality mapping (matching Python)
  SOURCE_QUALITY: { 'savills': 4, 'knightfrank': 4, 'knight frank': 4, 'chestertons': 3, 'foxtons': 2, 'rightmove': 1, 'zoopla': 1 },

  // Property type numeric mapping (matching Python)
  PROPERTY_TYPE_NUM: { 'studio': 0, 'flat': 1, 'apartment': 1, 'maisonette': 2, 'house': 3, 'penthouse': 4, 'townhouse': 3, 'town house': 3 },

  // Postcode frequency lookup (from training data distribution)
  POSTCODE_FREQ: {
    'SW3': 0.074, 'SW7': 0.068, 'W8': 0.055, 'W2': 0.052, 'SW5': 0.048,
    'SW11': 0.045, 'SW10': 0.044, 'NW8': 0.042, 'W11': 0.041, 'SW1X': 0.038,
    'NW3': 0.036, 'SW1W': 0.034, 'W14': 0.032, 'NW1': 0.030, 'W10': 0.028,
    'SW6': 0.025, 'W1': 0.024, 'SW1': 0.022, 'EC1': 0.020, 'WC1': 0.018,
    'default': 0.015
  },

  // Postcode area frequency lookup (from training data)
  POSTCODE_AREA_FREQ: {
    'SW': 0.44, 'W': 0.28, 'NW': 0.15, 'EC': 0.04, 'WC': 0.03,
    'E': 0.02, 'SE': 0.02, 'N': 0.01, 'default': 0.01
  },

  // Size quintile boundaries (from training data)
  SIZE_QUINTILES: [0, 484, 635, 818, 1141, Infinity],  // Q1-Q5 boundaries

  // Floor patterns matching Python FloorplanExtractor (canonical_name -> regex patterns)
  FLOOR_PATTERNS: {
    'basement': [/basement/i, /cellar/i],
    'lower_ground': [/lower\s*ground\s*(?:floor)?/i, /lgf\b/i, /lower\s*level/i],
    'ground': [/ground\s*(?:floor)?/i, /\bgf\b/i, /street\s*level/i, /raised\s*ground\s*(?:floor)?/i],
    'mezzanine': [/mezzanine/i, /mezz\b/i],
    'first': [/first\s*(?:floor)?/i, /1st\s*(?:floor)?/i],
    'second': [/second\s*(?:floor)?/i, /2nd\s*(?:floor)?/i],
    'third': [/third\s*(?:floor)?/i, /3rd\s*(?:floor)?/i],
    'fourth': [/fourth\s*(?:floor)?/i, /4th\s*(?:floor)?/i],
    'fifth': [/fifth\s*(?:floor)?/i, /5th\s*(?:floor)?/i],
    'sixth': [/sixth\s*(?:floor)?/i, /6th\s*(?:floor)?/i],
    'seventh': [/seventh\s*(?:floor)?/i, /7th\s*(?:floor)?/i],
    'eighth': [/eighth\s*(?:floor)?/i, /8th\s*(?:floor)?/i],
    'ninth': [/ninth\s*(?:floor)?/i, /9th\s*(?:floor)?/i],
    'tenth': [/tenth\s*(?:floor)?/i, /10th\s*(?:floor)?/i],
    'penthouse': [/penthouse/i],
    'roof_terrace': [/roof\s*terrace/i, /rooftop/i],
  },

  // Normalize OCR text (fix common typos)
  normalizeOcrText(text) {
    if (!text) return '';
    return text
      .replace(/\bflocr\b/gi, 'floor')
      .replace(/\bfioor\b/gi, 'floor')
      .replace(/\bFloor\s*Flan\b/gi, 'Floor Plan')
      .replace(/\b2nth\b/gi, '2nd')
      .replace(/\b17th\s*Flocr\b/gi, '17th Floor');
  },

  // Filter out exclusion lines (e.g., "excluding basement")
  filterExclusionLines(text) {
    if (!text) return '';
    const exclusionPattern = /(?:excluding|exclude|not\s+included|reduced\s+headroom)/i;
    return text.split('\n')
      .filter(line => !exclusionPattern.test(line))
      .join('\n');
  },

  // Extract floor information from OCR text (matches Python FloorplanExtractor._classify_floors)
  extractFloors(ocrText) {
    const result = {
      has_basement: 0,
      has_ground: 0,
      has_first_floor: 0,
      has_second_floor: 0,
      has_third_floor: 0,
      has_fourth_plus: 0,
      has_roof_terrace: 0,
      floor_count: 0,
      is_multi_floor: 0,
      floors_detected: [],
    };

    if (!ocrText) return result;

    // Normalize and filter text
    const normalizedText = this.normalizeOcrText(ocrText);
    const filteredText = this.filterExclusionLines(normalizedText);
    const textLower = filteredText.toLowerCase();

    // Track canonical floors found
    const canonicalFloors = new Set();

    // Match floor patterns
    for (const [canonical, patterns] of Object.entries(this.FLOOR_PATTERNS)) {
      for (const pattern of patterns) {
        if (pattern.test(textLower)) {
          canonicalFloors.add(canonical);
          break;
        }
      }
    }

    // Set binary flags based on canonical floor names
    // IMPORTANT: Lower ground floor is treated like basement (below street level = less desirable)
    // This matches Python: features['is_basement'] = int(is_basement or is_lower_ground)
    if (canonicalFloors.has('basement') || canonicalFloors.has('lower_ground')) {
      result.has_basement = 1;
      result.floors_detected.push(canonicalFloors.has('basement') ? 'basement' : 'lower_ground');
    }
    if (canonicalFloors.has('ground')) {
      result.has_ground = 1;
      result.floors_detected.push('ground');
    }
    if (canonicalFloors.has('first') || canonicalFloors.has('mezzanine')) {
      result.has_first_floor = 1;
      result.floors_detected.push('first');
    }
    if (canonicalFloors.has('second')) {
      result.has_second_floor = 1;
      result.floors_detected.push('second');
    }
    if (canonicalFloors.has('third')) {
      result.has_third_floor = 1;
      result.floors_detected.push('third');
    }
    // Fourth floor and above (including penthouse)
    const fourthPlusFloors = ['fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'penthouse'];
    if (fourthPlusFloors.some(f => canonicalFloors.has(f))) {
      result.has_fourth_plus = 1;
      result.floors_detected.push('fourth+');
    }
    if (canonicalFloors.has('roof_terrace')) {
      result.has_roof_terrace = 1;
      result.floors_detected.push('roof_terrace');
    }

    // Count unique main floors (excluding roof_terrace)
    const mainFloors = ['basement', 'lower_ground', 'ground', 'mezzanine', 'first', 'second', 'third',
                        'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'penthouse'];
    result.floor_count = mainFloors.filter(f => canonicalFloors.has(f)).length;

    // is_multi_floor = floor_count >= 2
    result.is_multi_floor = result.floor_count >= 2 ? 1 : 0;

    console.log(`[XGB] Floors extracted: ${result.floors_detected.join(', ') || 'none'}, count=${result.floor_count}, multi=${result.is_multi_floor}`);

    return result;
  },

  haversine(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const toRad = x => x * Math.PI / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
    return R * 2 * Math.asin(Math.sqrt(a));
  },

  // Detect premium agent from agent name or URL (matching Python)
  isPremiumAgent(agentName, pageUrl) {
    const combined = ((agentName || '') + ' ' + (pageUrl || '')).toLowerCase();
    return this.PREMIUM_AGENTS.some(agent => combined.includes(agent)) ? 1 : 0;
  },

  // Get source quality from URL (matching Python)
  getSourceQuality(pageUrl) {
    if (!pageUrl) return 1;
    const urlLower = pageUrl.toLowerCase();
    for (const [source, quality] of Object.entries(this.SOURCE_QUALITY)) {
      if (urlLower.includes(source.replace(' ', ''))) return quality;
    }
    return 1; // default for unknown sources
  },

  // Get property type numeric encoding (matching Python)
  getPropertyTypeNum(propertyType) {
    const typeLower = (propertyType || 'flat').toLowerCase();
    return this.PROPERTY_TYPE_NUM[typeLower] ?? 1;
  },

  // Get postcode frequency from lookup (matching Python training data)
  getPostcodeFreq(postcodeDistrict) {
    return this.POSTCODE_FREQ[postcodeDistrict] || this.POSTCODE_FREQ['default'];
  },

  // Get postcode area frequency (matching Python)
  getPostcodeAreaFreq(postcodeDistrict) {
    const area = postcodeDistrict.match(/^([A-Z]+)/i)?.[1]?.toUpperCase() || 'SW';
    return this.POSTCODE_AREA_FREQ[area] || this.POSTCODE_AREA_FREQ['default'];
  },

  // Get size bin (quintile) from sqft (matching Python pd.qcut)
  getSizeBin(sqft) {
    for (let i = 0; i < this.SIZE_QUINTILES.length - 1; i++) {
      if (sqft < this.SIZE_QUINTILES[i + 1]) return i;
    }
    return 4;
  },

  // Detect short let from description/URL (matching Python)
  isShortLet(description, pageUrl) {
    const combined = ((description || '') + ' ' + (pageUrl || '')).toLowerCase();
    return (combined.includes('short let') || combined.includes('short-let') ||
            combined.includes('holiday let') || combined.includes('serviced apartment') ||
            combined.includes('corporate let')) ? 1 : 0;
  },

  // === V17 PROPERTY TYPE DETECTION ===
  // V19: Check if property is a mews (from type or address)
  isMews(propertyType, address) {
    if (!propertyType && !address) return 0;
    const typeLower = (propertyType || '').toLowerCase();
    const addrLower = (address || '').toLowerCase();
    return (typeLower.includes('mews') || addrLower.includes('mews')) ? 1 : 0;
  },

  // Check if property type is a house variant (V19: excludes mews)
  isHouseType(propertyType) {
    if (!propertyType) return 0;
    const typeLower = propertyType.toLowerCase();
    // V19: Mews are NOT houses - they have distinct pricing
    if (typeLower.includes('mews')) return 0;
    return this.HOUSE_TYPES.some(h => typeLower.includes(h)) ? 1 : 0;
  },

  // Check if property type is a flat variant
  isFlatType(propertyType) {
    if (!propertyType) return 0;
    const typeLower = propertyType.toLowerCase();
    return this.FLAT_TYPES.some(f => typeLower.includes(f)) ? 1 : 0;
  },

  // Get type-based PPSF target (overall median for type) - V19: handles mews
  getTypePpsfTarget(isHouse, isFlat, isMews) {
    if (isMews) return this.OVERALL_MEWS_PPSF;
    if (isHouse) return this.OVERALL_HOUSE_PPSF;
    if (isFlat) return this.OVERALL_FLAT_PPSF;
    return this.OVERALL_PPSF;
  },

  // Get postcode × type PPSF (more granular) - V19: handles mews
  getPcTypePpsf(postcodeDistrict, isHouse, isFlat, isMews) {
    if (isMews) {
      return this.PC_MEWS_PPSF[postcodeDistrict] || this.PC_MEWS_PPSF['default'];
    }
    if (isHouse) {
      return this.PC_HOUSE_PPSF[postcodeDistrict] || this.PC_HOUSE_PPSF['default'];
    }
    if (isFlat) {
      return this.PC_FLAT_PPSF[postcodeDistrict] || this.PC_FLAT_PPSF['default'];
    }
    return this.OVERALL_PPSF;
  },

  // === V18 FURNISHED DETECTION ===
  // Detect furnished status from description
  detectFurnishedStatus(description) {
    if (!description) return { furnished: 0, unfurnished: 0, partFurnished: 0 };
    const desc = description.toLowerCase();

    // Check for unfurnished first (more specific)
    if (desc.includes('unfurnished')) {
      return { furnished: 0, unfurnished: 1, partFurnished: 0 };
    }
    // Check for part-furnished
    if (desc.includes('part furnished') || desc.includes('part-furnished')) {
      return { furnished: 0, unfurnished: 0, partFurnished: 1 };
    }
    // Check for furnished
    if (desc.includes('furnished')) {
      return { furnished: 1, unfurnished: 0, partFurnished: 0 };
    }
    return { furnished: 0, unfurnished: 0, partFurnished: 0 };
  },

  // === V16/V20 SOCIAL HOUSING DETECTION ===
  // Detect social housing from address and PPSF
  isSocialHousing(address, ppsf, postcodeDistrict) {
    if (!address) return 0;
    const addrLower = address.toLowerCase();

    // High confidence: Known estate names
    for (const pattern of this.SOCIAL_ESTATE_PATTERNS) {
      if (pattern.test(addrLower)) return 1;
    }

    // Medium confidence: Premium postcode + very low PPSF
    if (ppsf && ppsf < 3.5) {
      if (this.PREMIUM_POSTCODES_SOCIAL.some(p => postcodeDistrict.startsWith(p))) {
        return 1;
      }
    }

    return 0;
  },

  // Get postcode adjustment factor (V16)
  getPostcodeAdjustment(postcodeDistrict) {
    return this.POSTCODE_ADJUSTMENTS[postcodeDistrict] || this.POSTCODE_ADJUSTMENTS['default'];
  },

  // === V16 SIZE ANOMALY DETECTION ===
  isTiny(sqft) {
    return sqft < 400 ? 1 : 0;
  },

  isHuge(sqft) {
    return sqft >= 3000 ? 1 : 0;
  },

  // === V16 BATHROOM LUXURY SIGNALS ===
  hasEnsuiteEach(bathrooms, bedrooms) {
    const bedsAdj = Math.max(bedrooms, 0.5);
    return (bathrooms / bedsAdj) >= 1 ? 1 : 0;
  },

  highBathroomCount(bathrooms) {
    return bathrooms >= 4 ? 1 : 0;
  },

  excessBathrooms(bathrooms, bedrooms) {
    return Math.max(0, bathrooms - bedrooms);
  },

  // === V16 PROPERTY TYPE REFINEMENTS ===
  isTerraced(propertyType) {
    if (!propertyType) return 0;
    const typeLower = propertyType.toLowerCase();
    return (typeLower.includes('terraced') || typeLower.includes('town house')) ? 1 : 0;
  },

  isStudio(bedrooms, sqft) {
    return (bedrooms === 1 && sqft < 400) ? 1 : 0;
  },

  isHouseboat(propertyType) {
    if (!propertyType) return 0;
    return propertyType.toLowerCase().includes('house boat') ? 1 : 0;
  },

  isDuplexMaisonette(propertyType) {
    if (!propertyType) return 0;
    const typeLower = propertyType.toLowerCase();
    return (typeLower.includes('duplex') || typeLower.includes('maisonette')) ? 1 : 0;
  },

  isPenthouse(propertyType, address) {
    if (!propertyType && !address) return 0;
    const typeLower = (propertyType || '').toLowerCase();
    const addrLower = (address || '').toLowerCase();
    return (typeLower === 'penthouse' || addrLower.includes('penthouse')) ? 1 : 0;
  },

  // === V16 FLOOR TYPE KEYWORDS ===
  isGardenFlat(address) {
    if (!address) return 0;
    return address.toLowerCase().includes('garden flat') ? 1 : 0;
  },

  isBasementFlat(address) {
    if (!address) return 0;
    const addrLower = address.toLowerCase();
    return (addrLower.includes('basement flat') || addrLower.includes('basement apartment')) ? 1 : 0;
  },

  isGroundFloor(hasGround) {
    return hasGround === 1 ? 1 : 0;
  },

  // === V16 CONDITION DETECTION ===
  hasRefurbKeywords(description) {
    if (!description) return 0;
    const descLower = description.toLowerCase();
    const keywords = ['refurbished', 'newly decorated', 'newly renovated',
                      'brand new', 'just completed', 'newly fitted'];
    return keywords.some(kw => descLower.includes(kw)) ? 1 : 0;
  },

  // === V16 ADDRESS PREMIUM DETECTION ===
  // Detect garden square from address
  isGardenSquare(address) {
    if (!address) return 0;
    const addrLower = address.toLowerCase();
    return this.GARDEN_SQUARES.some(sq => addrLower.includes(sq)) ? 1 : 0;
  },

  // Detect ultra-prime address
  isUltraPrimeAddress(address) {
    if (!address) return 0;
    const addrLower = address.toLowerCase();
    return this.ULTRA_PRIME_ADDRESSES.some(addr => addrLower.includes(addr)) ? 1 : 0;
  },

  // Detect prime street
  isPrimeStreet(address) {
    if (!address) return 0;
    const addrLower = address.toLowerCase();
    return this.PRIME_STREETS.some(st => addrLower.includes(st)) ? 1 : 0;
  },

  // Calculate address prestige score (0-3 based on features)
  getAddressPrestige(isGardenSquare, isUltraPrime, isPrimeStreet) {
    return isUltraPrime * 3 + isGardenSquare * 2 + isPrimeStreet;
  },

  // === V20+ PRESTIGE LOCATION FUNCTIONS ===
  // Find the prestige location in address
  getPrestigeLocation(address) {
    if (!address) return null;
    const addrLower = address.toLowerCase();
    for (const location of Object.keys(this.PRESTIGE_LOCATION_PPSF)) {
      if (location !== 'default' && addrLower.includes(location)) {
        return location;
      }
    }
    return null;
  },

  // Get location-specific PPSF target (V20+)
  getPrestigeLocationPpsf(address) {
    const location = this.getPrestigeLocation(address);
    if (location) {
      return this.PRESTIGE_LOCATION_PPSF[location] || this.PRESTIGE_LOCATION_PPSF['default'];
    }
    return this.PRESTIGE_LOCATION_PPSF['default'];
  },

  // Get prestige tier (0-4) for address (V20+)
  getPrestigeTier(address) {
    const location = this.getPrestigeLocation(address);
    if (location) {
      return this.PRESTIGE_LOCATION_TIER[location] || 0;
    }
    return 0;
  },

  // Get prestige multiplier based on tier (V20+)
  getPrestigeMultiplier(address) {
    const tier = this.getPrestigeTier(address);
    return this.PRESTIGE_TIER_MULTIPLIER[tier] || 1.0;
  },

  // Parse amenities from description (matching Python exactly)
  parseAmenities(text) {
    if (!text) return {};
    const t = text.toLowerCase();
    return {
      has_balcony: t.includes('balcony') ? 1 : 0,
      has_terrace: (t.includes('terrace') && !t.includes('roof terrace')) ? 1 : 0,
      has_roof_terrace: t.includes('roof terrace') ? 1 : 0,
      has_garden: t.includes('garden') ? 1 : 0,
      has_porter: (t.includes('porter') || t.includes('concierge')) ? 1 : 0,
      has_gym: (t.includes('gym') || t.includes('fitness')) ? 1 : 0,
      has_pool: (t.includes('pool') || t.includes('swimming')) ? 1 : 0,  // Added 'swimming'
      has_parking: (t.includes('parking') || t.includes('garage')) ? 1 : 0,  // Added 'garage'
      has_lift: (t.includes('lift') || t.includes('elevator')) ? 1 : 0,
      has_ac: (t.includes('air con') || t.includes('a/c') || t.includes('air-con') || t.includes('aircon')) ? 1 : 0,  // Added 'aircon'
      has_high_ceilings: t.includes('high ceiling') ? 1 : 0,
      has_view: t.includes('view') ? 1 : 0,  // Removed space prefix to match Python
      has_modern: (t.includes('modern') || t.includes('contemporary')) ? 1 : 0,  // Added 'contemporary'
      has_period: (t.includes('period') || t.includes('victorian') || t.includes('georgian')) ? 1 : 0,  // Added 'georgian'
      has_furnished: (t.includes('furnished') && !t.includes('unfurnished')) ? 1 : 0,
    };
  },

  // Extract postcode district properly (handles SW1X, SW1W, NW3, W8, etc.)
  extractPostcodeDistrict(postcode) {
    if (!postcode) return 'SW3';
    // Get outcode (part before space)
    const outcode = postcode.split(' ')[0].toUpperCase();
    // For districts like SW1X, SW1W, W1K - keep the full outcode
    // Match pattern: letters + digits + optional letter (e.g., SW1X, NW3, W8, EC1)
    const match = outcode.match(/^([A-Z]{1,2}\d{1,2}[A-Z]?)$/);
    return match ? match[1] : outcode;
  },

  // Generate postcode one-hot encoding
  getPostcodeOneHot(postcodeDistrict) {
    const result = {};
    for (const pc of this.POSTCODE_FEATURES) {
      const district = pc.replace('pc_', '');
      result[pc] = (postcodeDistrict === district) ? 1 : 0;
    }
    return result;
  },

  // Generate property type one-hot encoding
  getPropertyTypeOneHot(propertyType) {
    const result = {};
    const normalizedType = (propertyType || '').toLowerCase().trim();

    // Initialize all to 0
    for (const typeFeature of this.PROPERTY_TYPE_FEATURES) {
      result[typeFeature] = 0;
    }

    // Priority matching - check specific types first to avoid substring issues
    // e.g., "penthouse" should match type_penthouse, not type_house
    // Order matters! More specific patterns must come first
    const typeMapping = {
      // Most specific first
      'house of multiple occupation': 'type_house of multiple occupation',
      'link detached house': 'type_link detached house',
      'link detached': 'type_link detached house',
      'ground floor flat': 'type_ground flat',
      'ground flat': 'type_ground flat',
      'end of terrace': 'type_end of terrace',
      'semi-detached': 'type_semi-detached',
      'house share': 'type_house share',
      'house boat': 'type_house boat',
      'houseboat': 'type_house boat',
      'town house': 'type_town house',
      'townhouse': 'type_town house',
      'long let': 'type_long let',
      // Medium specificity
      'penthouse': 'type_penthouse',
      'maisonette': 'type_maisonette',
      'terraced': 'type_terraced',
      'detached': 'type_detached',
      'apartment': 'type_apartment',
      'studio': 'type_studio',
      'duplex': 'type_duplex',
      'parking': 'type_parking',
      'mews': 'type_mews',
      'flat': 'type_flat',
      // Least specific (check last)
      'house': 'type_house',
    };

    // Find best match (check longer/more specific terms first)
    let matched = false;
    for (const [keyword, featureName] of Object.entries(typeMapping)) {
      if (normalizedType.includes(keyword)) {
        if (result.hasOwnProperty(featureName)) {
          result[featureName] = 1;
          matched = true;
          break; // Stop at first match (most specific due to order)
        }
      }
    }

    // Default to flat if no match
    if (!matched) {
      result['type_flat'] = 1;
    }

    return result;
  },

  buildFeatures(data) {
    const beds = data.bedrooms || 1;
    const baths = data.bathrooms || 1;
    const sqft = data.size_sqft || (beds * 450);
    const postcode = data.postcode || 'SW3';
    const postcodeDistrict = this.extractPostcodeDistrict(postcode);
    const propertyType = data.propertyType || 'flat';
    const lat = data.latitude || this.CITY_CENTER.lat;
    const lon = data.longitude || this.CITY_CENTER.lon;
    const description = data.description || '';
    const ocrText = data.ocrText || '';
    const agentName = data.agentName || '';
    const pageUrl = data.pageUrl || window.location.href;
    const address = data.address || '';  // V16: need address for premium detection

    // Calculate distances
    const tubeDist = Math.min(...Object.values(this.TUBE_STATIONS).map(
      s => this.haversine(lat, lon, s.lat, s.lon)
    ));
    const centerDist = this.haversine(lat, lon, this.CITY_CENTER.lat, this.CITY_CENTER.lon);
    const centerInv = 1 / (1 + centerDist);

    // Parse amenities from description
    const amenities = this.parseAmenities(description);
    const amenityScore = Object.values(amenities).reduce((a, b) => a + b, 0);
    const isPrime = this.PRIME_POSTCODES.some(p => postcodeDistrict.startsWith(p)) ? 1 : 0;

    const hasOutdoor = (amenities.has_balcony || amenities.has_terrace || amenities.has_garden || amenities.has_roof_terrace) ? 1 : 0;
    const premiumAmenityCount = amenities.has_pool + amenities.has_porter + amenities.has_gym + amenities.has_ac;

    // Extract floor information from OCR text (matches Python FloorplanExtractor)
    const floors = this.extractFloors(ocrText);

    // Generate one-hot encodings
    const postcodeOneHot = this.getPostcodeOneHot(postcodeDistrict);
    const propertyTypeOneHot = this.getPropertyTypeOneHot(propertyType);

    // Calculate dynamic features (previously hardcoded)
    const isPremiumAgent = this.isPremiumAgent(agentName, pageUrl);
    const sourceQuality = this.getSourceQuality(pageUrl);
    const propertyTypeNum = this.getPropertyTypeNum(propertyType);
    const postcodeFreq = this.getPostcodeFreq(postcodeDistrict);
    const postcodeAreaFreq = this.getPostcodeAreaFreq(postcodeDistrict);
    const sizeBin = this.getSizeBin(sqft);
    const shortLet = this.isShortLet(description, pageUrl);
    const longLet = shortLet ? 0 : 1;

    // === V16/V20 ADDRESS-BASED PREMIUM FEATURES ===
    const isGardenSquare = this.isGardenSquare(address);
    const isUltraPrimeAddress = this.isUltraPrimeAddress(address);
    const isPrimeStreet = this.isPrimeStreet(address);
    const addressPrestige = this.getAddressPrestige(isGardenSquare, isUltraPrimeAddress, isPrimeStreet);

    // === V20+ PRESTIGE LOCATION FEATURES (more accurate than binary) ===
    const prestigeLocationPpsf = this.getPrestigeLocationPpsf(address);
    const prestigeTier = this.getPrestigeTier(address);
    const prestigeMultiplier = this.getPrestigeMultiplier(address);
    const prestigeExpectedPrice = prestigeLocationPpsf * sqft;
    const logPrestigeExpectedPrice = Math.log1p(prestigeExpectedPrice);
    const prestigePpsfRatio = prestigeLocationPpsf / this.PRESTIGE_LOCATION_PPSF['default'];

    // === V16/V20 NEW FEATURES ===
    // Calculate PPSF for social housing detection (if price available)
    const ppsf = data.price_pcm ? data.price_pcm / sqft : null;
    const isSocialHousing = this.isSocialHousing(address, ppsf, postcodeDistrict);
    const postcodeAdjustment = this.getPostcodeAdjustment(postcodeDistrict);
    const isUltraLuxuryAddress = isUltraPrimeAddress; // V16 uses same detection
    const isTiny = this.isTiny(sqft);
    const isHuge = this.isHuge(sqft);
    const hasEnsuiteEach = this.hasEnsuiteEach(baths, beds);
    const highBathroomCount = this.highBathroomCount(baths);
    const excessBathrooms = this.excessBathrooms(baths, beds);
    const isTerraced = this.isTerraced(propertyType);
    const isStudio = this.isStudio(beds, sqft);
    const isHouseboat = this.isHouseboat(propertyType);
    const isDuplexMaisonette = this.isDuplexMaisonette(propertyType);
    const isPenthouse = this.isPenthouse(propertyType, address);
    const isGardenFlat = this.isGardenFlat(address);
    const isBasementFlat = this.isBasementFlat(address);
    const hasRefurbKeywords = this.hasRefurbKeywords(description);

    // === V19 MEWS DETECTION (before V17 type detection) ===
    const isMews = this.isMews(propertyType, address);

    // === V17 PROPERTY TYPE × SIZE FEATURES ===
    // V19: isHouseType now excludes mews
    const isHouse = this.isHouseType(propertyType);
    const isFlat = this.isFlatType(propertyType);
    const isLargeHouse = (isHouse && sqft > 2000) ? 1 : 0;
    // V19: Pass isMews to get correct PPSF
    const typePpsfTarget = this.getTypePpsfTarget(isHouse, isFlat, isMews);
    const pcTypePpsf = this.getPcTypePpsf(postcodeDistrict, isHouse, isFlat, isMews);
    const typeExpectedPrice = typePpsfTarget * sqft;
    const pcTypeExpectedPrice = pcTypePpsf * sqft;

    // === V18 FURNISHED FEATURES ===
    const furnishedStatus = this.detectFurnishedStatus(description);
    const isFurnishedExplicit = furnishedStatus.furnished;
    const isUnfurnished = furnishedStatus.unfurnished;
    const isPartFurnished = furnishedStatus.partFurnished;

    // Log feature extraction
    console.log(`[XGB] Postcode district: ${postcodeDistrict}, one-hot match: ${Object.entries(postcodeOneHot).find(([k,v]) => v===1)?.[0] || 'none'}`);
    console.log(`[XGB] Property type: ${propertyType}, one-hot match: ${Object.entries(propertyTypeOneHot).find(([k,v]) => v===1)?.[0] || 'none'}`);
    console.log(`[XGB] Dynamic features: premium_agent=${isPremiumAgent}, source_quality=${sourceQuality}, size_bin=${sizeBin}, short_let=${shortLet}`);
    console.log(`[XGB] Postcode freq: ${postcodeFreq.toFixed(3)}, area_freq: ${postcodeAreaFreq.toFixed(2)}`);
    console.log(`[XGB] V16/V20: social_housing=${isSocialHousing}, pc_adj=${postcodeAdjustment.toFixed(2)}, tiny=${isTiny}, huge=${isHuge}`);
    console.log(`[XGB] V16/V20: high_bath=${highBathroomCount}, terraced=${isTerraced}, penthouse=${isPenthouse}, refurb=${hasRefurbKeywords}`);
    console.log(`[XGB] V16 Address: garden_sq=${isGardenSquare}, ultra_prime=${isUltraPrimeAddress}, prime_st=${isPrimeStreet}, prestige=${addressPrestige}`);
    console.log(`[XGB] V17 Type: is_house=${isHouse}, is_flat=${isFlat}, is_large_house=${isLargeHouse}, pc_type_ppsf=${pcTypePpsf.toFixed(2)}`);
    console.log(`[XGB] V18 Furnished: explicit=${isFurnishedExplicit}, unfurnished=${isUnfurnished}, part=${isPartFurnished}`);
    console.log(`[XGB] V19 Mews: is_mews=${isMews}${isMews ? `, pc_mews_ppsf=${pcTypePpsf.toFixed(2)}` : ''}`);
    console.log(`[XGB] V20 Prestige: tier=${prestigeTier}, ppsf=${prestigeLocationPpsf.toFixed(2)}, multiplier=${prestigeMultiplier.toFixed(2)}, ratio=${prestigePpsfRatio.toFixed(2)}`);

    // Calculate floor-related features (matching Python rental_price_models_v15.py)
    const floorCount = floors.floor_count || 1;
    const floorSizeInteraction = floorCount * sqft / 1000;
    const logSqft = Math.log1p(sqft);

    // Floor type from keywords
    const isGroundFloor = floors.has_ground;

    return {
      // === CORE FEATURES ===
      bedrooms: beds,
      bathrooms: baths,
      size_sqft: sqft,
      size_per_bed: sqft / Math.max(beds, 0.5),
      bed_bath_interaction: beds * baths,
      log_sqft: logSqft,
      sqrt_sqft: Math.sqrt(sqft),
      size_squared: sqft ** 2 / 100000,
      beds_squared: beds ** 2,
      size_bin: sizeBin,

      // === V16/V20 SIZE ANOMALIES ===
      is_tiny: isTiny,
      is_huge: isHuge,

      // === V16/V20 BATHROOM FEATURES ===
      bath_ratio: baths / Math.max(beds, 0.5),
      has_ensuite_each: hasEnsuiteEach,
      high_bathroom_count: highBathroomCount,
      excess_bathrooms: excessBathrooms,

      // === LOCATION FEATURES ===
      tube_distance_km: tubeDist,
      log_tube_distance: Math.log1p(tubeDist),
      center_distance_km: centerDist,
      log_center_distance: Math.log1p(centerDist),
      center_distance_inv: centerInv,
      is_prime_postcode: isPrime,
      postcode_freq: postcodeFreq,
      postcode_area_freq: postcodeAreaFreq,
      postcode_adjustment: postcodeAdjustment,

      // === V16/V20 DETECTION FEATURES ===
      is_social_housing: isSocialHousing,
      is_ultra_luxury_address: isUltraLuxuryAddress,

      // === V16/V19 ADDRESS PREMIUM FEATURES ===
      is_garden_square: isGardenSquare,
      is_ultra_prime_address: isUltraPrimeAddress,
      is_prime_street: isPrimeStreet,
      address_prestige: addressPrestige,

      // === V20+ PRESTIGE LOCATION FEATURES ===
      prestige_location_ppsf: prestigeLocationPpsf,
      prestige_tier: prestigeTier,
      prestige_multiplier: prestigeMultiplier,
      log_prestige_expected_price: logPrestigeExpectedPrice,
      prestige_ppsf_ratio: prestigePpsfRatio,

      // === V19 MEWS FEATURES ===
      is_mews: isMews,

      // === V16/V17 PROPERTY TYPE FEATURES ===
      is_house: isHouse,
      is_flat: isFlat,
      is_large_house: isLargeHouse,
      is_terraced: isTerraced,
      is_penthouse: isPenthouse,
      is_studio: isStudio,
      is_houseboat: isHouseboat,
      is_duplex_maisonette: isDuplexMaisonette,
      property_type_num: propertyTypeNum,

      // === V17/V19 TYPE TARGET ENCODING ===
      type_ppsf_target: typePpsfTarget,
      log_type_expected_price: Math.log1p(typeExpectedPrice),
      pc_type_ppsf: pcTypePpsf,
      log_pc_type_expected_price: Math.log1p(pcTypeExpectedPrice),

      // === FLOOR FEATURES ===
      floor_count: floorCount,
      is_multi_floor: floors.is_multi_floor,
      floor_size_interaction: floorSizeInteraction,
      has_basement: floors.has_basement,
      has_ground: floors.has_ground,
      has_first_floor: floors.has_first_floor,
      has_second_floor: floors.has_second_floor,
      has_third_floor: floors.has_third_floor,
      has_fourth_plus: floors.has_fourth_plus,
      is_garden_flat: isGardenFlat,
      is_basement_flat: isBasementFlat,
      is_ground_floor: isGroundFloor,

      // === V16/V18 CONDITION FEATURES ===
      is_furnished_explicit: isFurnishedExplicit,
      is_unfurnished: isUnfurnished,
      is_part_furnished: isPartFurnished,
      has_refurb_keywords: hasRefurbKeywords,
      is_long_let: longLet,
      is_short_let: shortLet,

      // === AGENT FEATURES ===
      is_premium_agent: isPremiumAgent,
      premium_agent_size: isPremiumAgent * logSqft,
      source_quality: sourceQuality,

      // === AMENITY FEATURES ===
      amenity_score: amenityScore,
      premium_amenity_count: premiumAmenityCount,
      has_outdoor_space: hasOutdoor,
      amenity_x_central: amenityScore * centerInv,
      outdoor_x_prime: hasOutdoor * isPrime,

      // === INTERACTION FEATURES ===
      size_x_central: sqft * centerInv / 100,
      size_x_prime: sqft * isPrime / 1000,
      beds_x_central: beds * centerInv,
      garden_square_size: isGardenSquare * logSqft,
      ultra_prime_size: isUltraPrimeAddress * sqft / 1000,
      prime_street_size: isPrimeStreet * logSqft,
      prestige_x_size: addressPrestige * logSqft,
      // V20+ prestige location interactions (more accurate than binary)
      prestige_tier_x_size: prestigeTier * logSqft,
      prestige_ppsf_x_sqft: prestigeLocationPpsf * sqft / 1000,
      prestige_multiplier_x_type: prestigeMultiplier * typePpsfTarget,
      house_size_interaction: isHouse * logSqft,
      flat_size_interaction: isFlat * logSqft,
      large_house_size: isLargeHouse * sqft / 1000,
      mews_size_interaction: isMews * logSqft,
      mews_x_prime: isMews * isPrime,
      furnished_x_prime: isFurnishedExplicit * isPrime,
      furnished_x_central: isFurnishedExplicit * centerInv,
      unfurnished_discount: isUnfurnished * logSqft,
      luxury_address_size: isUltraLuxuryAddress * logSqft,
      luxury_bathroom_size: highBathroomCount * logSqft,
      penthouse_size: isPenthouse * logSqft,
      terraced_location: isTerraced * isPrime,
      short_let_x_central: shortLet * centerInv,
      short_let_size: shortLet * logSqft,

      // === AMENITY ONE-HOT ===
      ...amenities,

      // === ONE-HOT ENCODINGS ===
      ...postcodeOneHot,
      ...propertyTypeOneHot,
    };
  }
};

// Export for use in content.js
window.XGBoostPredictor = XGBoostPredictor;
window.XGBFeatures = XGBFeatures;
